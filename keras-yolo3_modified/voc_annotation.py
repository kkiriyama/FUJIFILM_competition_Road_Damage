import xml.etree.ElementTree as ET
import os
from os import getcwd

sets=range(1, 8)

def convert_annotation(location_n, image_id, list_file):
    image_id = image_id.split('.')[0]
    in_file = open('../train/location%d/labels/%s.xml'%(location_n, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls_id = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

list_file = open('train.txt', 'w')
for location_n in sets:
    image_ids = os.listdir('./train/location%d/images'%(location_n))
    for image_id in image_ids:
        list_file.write('%s/train/location%s/images/%s'%(wd, location_n, image_id))
        convert_annotation(location_n, image_id, list_file)
        list_file.write('\n')

list_file.close()