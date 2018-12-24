import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import xml.dom.minidom

from tqdm import tqdm

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
            r_image, n_cls_list, b_box_list = yolo.detect_image(image)
            output_list.append([img_name_withext, n_cls_list, b_box_list])
            r_image.save('./output_image/raw_model1/output_%s.jpg'%(img_name))
    yolo.close_session()

    xml = generate_xml(output_list)
    f = open('answer1_YOLO.xml', 'w')
    f.write(xml)
    f.close()

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
            name.appendChild(dom.createTextNode(output[1][i] - 1))
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
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
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
        '--image', default=False, action="store_true",
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

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
