import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import xml.dom.minidom
import numpy as np
import cv2 as cv

from tqdm import tqdm


def calc_score(yolo):
    split = [0.4, 0.6]
    annotation_path = './train_scaled.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    lines = np.array(lines)
    val_data = lines[int(len(lines) * split[0]) : int(len(lines) * split[1])]
    F1_scores = []
    for data in tqdm(val_data):
        TP = 0
        split_list = data.split()
        path = split_list[0]
        box_list = []
        for box in split_list[1:]:
            box_list.append(box.split(','))
        image = Image.open(path)
        r_image, n_cls_list, b_box_list, score_list = yolo.detect_image(image)
        final_predict_counter = 0 
        for i, predict_class in enumerate(n_cls_list):
            predict_box = b_box_list[i]
            score = score_list[i]
            correct_currentclass_list = []
            for box in box_list:
                if int(box[4]) == int(predict_class) - 1:
                    correct_currentclass_list.append(box)
            for correct in correct_currentclass_list:
                c_xmin = int(correct[0])
                c_ymin = int(correct[1])
                c_xmax = int(correct[2])
                c_ymax = int(correct[3])
                c_center = [(c_xmin + c_xmax)/2, (c_ymin + c_ymax)/2]
                p_center = [(predict_box[0] + predict_box[2])/2, (predict_box[1] + predict_box[3])/2]
                c_h = c_ymax - c_ymin
                c_w = c_xmax - c_xmin
                p_h = predict_box[3] - predict_box[1]
                p_w = predict_box[2] - predict_box[0]
                if ((np.abs(c_center[0] - p_center[0] < (c_w/2 + p_w/2))) & (np.abs(c_center[1] - p_center[1]) < (c_h/2 + p_h/2))):
                    overlap_w = (c_w/2 + p_w/2) - np.abs(c_center[0] - p_center[0])
                    overlap_h = (c_h/2 + p_h/2) - np.abs(c_center[1] - p_center[1])
                    if (overlap_w * overlap_h > (c_h * c_w + p_h * p_w - overlap_h * overlap_w) * 0.5):
                        TP += 1
        FP = final_predict_counter - TP
        FN = len(box_list) - TP
        TP += 0.0000001
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * recall * precision / (recall + precision)
        F1_scores.append(F1_score)
    # yolo.close_session()
    print(np.mean(np.array(F1_scores)))
    return np.mean(np.array(F1_scores))
                    
def calc_score_wrapper(x):
    x = np.array(x).astype(np.float32)
    FLAGS.iou = x[1]
    FLAGS.score = x[0]
    return -1 * calc_score(YOLO(**vars(FLAGS)))


def calc_area(box):
    w = int(box[2]) - int(box[1])
    h = int(box[3]) - int(box[0])
    return w * h 

def scaling_box(box, scale):
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    s_xmin = int(xmin - (xmax - xmin) / 2 * scale)
    if s_xmin < 0:
        s_xmin = 0
    s_ymin = int(ymin - (ymax - ymin) / 2 * scale)
    if s_ymin < 0:
        s_ymin = 0
    s_xmax = int(xmax + (xmax - xmin) / 2 * scale)
    if s_xmax > 600:
        s_xmax = 600
    s_ymax = int(ymax + (ymax - ymin) / 2 * scale)
    if s_ymax > 600:
        s_ymax = 600
    prev_area = (xmax - xmin) * (ymax - ymin)
    post_area = (s_xmax - s_xmin) * (s_ymax - s_ymin)
    print(post_area/prev_area)
    return [s_xmin, s_ymin, s_xmax, s_ymax]

def detect_img(yolo):
    img_list = os.listdir('./test')
    output_list = []
    for img in tqdm(img_list):
        img_name_withext = os.path.basename(img)
        img_name = os.path.splitext(img_name_withext)[0]
        try:
            image = Image.open('./test/' + img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, n_cls_list, b_box_list, score_list = yolo.detect_image(image)
            new_b_box_list = []
            scale = 0.1
            for i, cls in enumerate(n_cls_list):
                if cls in ['1', '2', '3', '4']:
                    if calc_area(b_box_list[i]) > 0:
                        new_b_box_list.append(scaling_box(b_box_list[i], scale))
                    else:
                        new_b_box_list.append(b_box_list[i])
                elif cls in ['6','8']:
                    if calc_area(b_box_list[i]) > 0:
                        new_b_box_list.append(scaling_box(b_box_list[i], scale))
                    else:
                        new_b_box_list.append(b_box_list[i])
                else:
                    if calc_area(b_box_list[i]) > 0:
                        new_b_box_list.append(scaling_box(b_box_list[i], scale))
                    else:
                        new_b_box_list.append(b_box_list[i])
            assert len(n_cls_list) == len(new_b_box_list)
            output_list.append([img_name_withext, n_cls_list, new_b_box_list])
    yolo.close_session()
    return output_list

def generate_xml(output_list):
    dom = xml.dom.minidom.Document()

    root = dom.createElement('annotations')
    dom.appendChild(root)

    for output in output_list:
        annotation = dom.createElement('annotation')
        
        filename = dom.createElement('filename')
        filename.appendChild(dom.createTextNode(output[0]))
        
        for i in range(len(output[1])):

            obj = dom.createElement('object')

            name = dom.createElement('name')
            name.appendChild(dom.createTextNode(str(int(output[1][i]) - 1)))
            pose = dom.createElement('pose')
            pose.appendChild(dom.createTextNode('Unspecified'))
            truncated = dom.createElement('truncated')
            truncated.appendChild(dom.createTextNode('0'))
            difficult = dom.createElement('difficult')
            difficult.appendChild(dom.createTextNode('0'))

            bndbox = dom.createElement('bndbox')

            xmin = dom.createElement('xmin')
            xmin.appendChild(dom.createTextNode(str(output[2][i][0])))
            ymin = dom.createElement('ymin')
            ymin.appendChild(dom.createTextNode(str(output[2][i][1])))
            xmax = dom.createElement('xmax')
            xmax.appendChild(dom.createTextNode(str(output[2][i][2])))
            ymax = dom.createElement('ymax')
            ymax.appendChild(dom.createTextNode(str(output[2][i][3])))

            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)

            obj.appendChild(name)
            obj.appendChild(pose)
            obj.appendChild(truncated)
            obj.appendChild(difficult)
            obj.appendChild(bndbox)
            annotation.appendChild(obj)

        annotation.appendChild(filename)

        root.appendChild(annotation)
    
    return dom.toprettyxml()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--use_CV', type = bool
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    """
    if FLAGS.use_CV:
        rets = []
        for i in range(1):
            FLAGS.model = '../logs/000/trained_weights_final_%d.h5'%(i)
            output_list = detect_img(YOLO(**vars(FLAGS)))
            rets.append(output_list)


        dom = xml.dom.minidom.Document()
        root = dom.createElement('annotations')
        dom.appendChild(root)

        for i in range(len(rets[0])):
            annotation = dom.createElement('annotation')
            filename = dom.createElement('filename')
            annotation.appendChild(filename)
            for j in range(1, 9):
                img = np.zeros((600, 600))
                for k in range(5):
                    for m in range(rets[k][i][1]):
                        if (int(rets[k][i][1][m]) - 1 == j):
                            img[rets[k][i][2][m][0]:rets[k][i][2][m][2], rets[k][i][2][m][1]:rets[k][i][2][m][3]] += 1
                img[img < 3] = 0
                img[img >= 3] = 100
                dst, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                for i, contour in enumerate(contours):
                    if area < 500:
                        continue
                    obj = dom.createElement('object')
                    name = dom.createElement('name')
                    name.appendChild(dom.createTextNode(str(j - 1)))
                    area = cv.contourArea(contour)
                    x,y,w,h = cv.boundingRect(contour)

                    xmin = dom.createElement('xmin')
                    xmin.appendChild(dom.createTextNode(str(x)))
                    obj.appendChild(xmin)
                    ymin = dom.createElement('ymin')
                    ymin.appendChild(dom.createTextNode(str(x)))
                    obj.appendChild(ymin)
                    xmax = dom.createElement('xmax')
                    xmax.appendChild(dom.createTextNode(str(x)))
                    obj.appendChild(xmax)
                    ymax = dom.createElement('ymax')
                    ymax.appendChild(dom.createTextNode(str(x)))
                    obj.appendChild(ymax)
                    annotation.appendChild(obj)
            root.appendChild(annotation)
        
        f = open('YOLO_answer_3.xml', 'w')
        f.write(dom.toprettyxml())
        f.close()
    else:
        FLAGS.model = './logs/000/trained_weights_final_0.h5'
        output_list = detect_img(YOLO(**vars(FLAGS)))
        xml = generate_xml(output_list)
        f = open('YOLO_answer_3.xml', 'w')
        f.write(xml)
        f.close()
    """

    FLAGS.model = './preprocessed_final_0.h5'
    output_list = detect_img(YOLO(**vars(FLAGS)))
    xml = generate_xml(output_list)
    f = open('YOLO_answer_preprocessed.xml', 'w')
    f.write(xml)
    f.close()

    """
    FLAGS.model = './logs/001/trained_weights_final_0.h5'
    space = [
        (0.05, 0.15),
        (0.6, 0.8)
    ]
    x0 = [0.1, 0.7]
    res = gp_minimize(calc_score_wrapper, space, x0 = x0, n_calls = 20)
    print(-1 * res.fun)
    print(res.x)
    """
    

