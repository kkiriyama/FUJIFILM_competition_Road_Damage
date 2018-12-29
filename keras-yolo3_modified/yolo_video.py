import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import xml.dom.minidom
import numpy as np
import cv2 as cv

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
            print(n_cls_list)
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
        '--use_CV', default = False 
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
    if FLAGS.use_CV:
        rets = []
        for i in range(5):
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
        
        f = open('YOLO_answer_2.xml', 'w')
        f.write(dom.toprettyxml())
        f.close()
    else:
        FLAGS.model = '../logs/000/trained_weights_final.h5'
        output_list = detect_img(YOLO(**vars(FLAGS)))
        xml = generate_xml(output_list)
        f = open('YOLO_answer_scaled_9.xml', 'w')
        f.write(xml)
        f.close()




        

