"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import os


def _main():
    annotation_path = './train_scaled.txt'
    log_dir = './logs/000/'
    classes_path = './model_data/voc_classes.txt'
    anchors_path = './model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting

    models = []

    for i in range(5):
        if is_tiny_version:
            model = create_tiny_model(input_shape, anchors, num_classes,
                freeze_body=2, weights_path='model_data/yolo_tiny.h5')
        else:
            model = create_model(input_shape, anchors, num_classes,
                freeze_body=2, weights_path='model_data/yolo.h5') # make sure you know what you freeze
        models.append(model)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    split = [0, 0.2, 0.4, 0.6, 0.8, 1]
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    lines = np.array(lines)

    for i in range(5):
        train_split_data = np.split(lines, [int(len(lines) * split[i]), int(len(lines) * split[i + 1])])
        train_data = np.hstack((train_split_data[0], train_split_data[2]))
        val_data = lines[int(len(lines) * split[i]) : int(len(lines) * split[i + 1])]
        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if True:
            model.compile(optimizer=Adam(lr=1e-3), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            batch_size = 32
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(int(len(lines) * 0.9), int(len(lines) * 0.1), batch_size))
            model.fit_generator(data_generator_wrapper(train_data, batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, int(len(lines) * 0.8)//batch_size),
                    validation_data=data_generator_wrapper(val_data, batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, int(len(lines) * 0.2)//batch_size),
                    epochs=10,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint])
            model.save_weights(log_dir + 'trained_weights_stage_1_%d.h5'%(i))

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for k in range(len(model.layers)):
                model.layers[k].trainable = True
            model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = 8 # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(int(len(lines) * 0.9), int(len(lines) * 0.1), batch_size))
            model.fit_generator(data_generator_wrapper(train_data, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, int(len(lines) * 0.8)//batch_size),
                validation_data=data_generator_wrapper(val_data, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, int(len(lines) * 0.2)//batch_size),
                epochs=20,
                initial_epoch=5,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            model.save(log_dir + 'trained_weights_final_%d.h5'%(i))

        # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(
            weights_path,
            by_name=True,
            skip_mismatch=True
            )
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(
            weights_path,
            by_name=True,
            skip_mismatch=True
        )
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
