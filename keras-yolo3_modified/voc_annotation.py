import xml.etree.ElementTree as ET
import os
from os import getcwd

sets=range(1, 8)
scale = 0.05

def convert_annotation(location_n, image_id, list_file):
    image_id = image_id.split('.')[0]
    in_file = open('./train/location%d/labels/%s.xml'%(location_n, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls_id = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text) * 0.693
        ymin = int(xmlbox.find('ymin').text) * 0.693
        xmax = int(xmlbox.find('xmax').text) * 0.693
        ymax = int(xmlbox.find('ymax').text) * 0.693
        scaled_xmin = int((((xmin + xmax) / 2 - xmin) * 0.05) + xmin)
        scaled_ymin = int((((ymin + ymax) / 2 - ymin) * 0.05) + ymin)
        scaled_xmax = int(xmax - (((xmin + xmax) / 2 - xmin) * 0.05))
        scaled_ymax = int(ymax - (((ymin + ymax) / 2 - ymin) * 0.05))
        b = (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax)
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

list_file = open('train_scaled.txt', 'w')
for location_n in sets:
    image_ids = os.listdir('./train/location%d/images'%(location_n))
    for image_id in image_ids:
        list_file.write('%s/train/location%s/images/%s'%(wd, location_n, image_id))
        convert_annotation(location_n, image_id, list_file)
        list_file.write('\n')

list_file.close()