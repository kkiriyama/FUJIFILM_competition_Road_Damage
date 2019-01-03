import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

sets = range(1, 8)

def parse_xml(location_n, image_id):
  image_id = image_id.split('.')[0]
  in_file = open('./train/location%d/labels/%s.xml'%(location_n, image_id))
  tree=ET.parse(in_file)
  root = tree.getroot()
  return root

def count_annotation(root):
  c = 0
  counter = np.zeros(8)
  for obj in root.iter('object'):
    cls_id = int(obj.find('name').text)
    counter[cls_id - 1] += 1
    c += 1
  return c, counter

def calc_area(obj):
  bbnox = obj.find('bndbox')
  xmin = int(bbox.find('xmin').text)
  ymin = int(bbox.find('ymin').text)
  xmax = int(bbox.find('xmax').text)
  ymax = int(bbox.find('ymax').text)
  return (xmax - xmin) * (ymax - ymin)

n_boxes = []
n_class = np.zeros(8)
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []
area_list = []
area_class_list = []

for i in range(8):
  area_class_list.append([])

print(area_list)

for location_n in sets:
  image_ids = os.listdir('./train/location%d/images'%(location_n))
  for image_id in image_ids:
    root = parse_xml(location_n, image_id)
    n_box, n_cls = count_annotation(root)
    n_boxes.append(n_box)
    n_class += n_cls
    for obj in root.iter('object'):
      bbox = obj.find('bndbox')
      xmin_list.append(int(bbox.find('xmin').text))
      ymin_list.append(int(bbox.find('ymin').text))
      xmax_list.append(int(bbox.find('xmax').text))
      ymax_list.append(int(bbox.find('ymax').text))
      area_list.append(calc_area(obj))
      class_id = int(obj.find('name').text)
      area_class_list[class_id - 1].append(calc_area(obj))


print(max(n_boxes), min(n_boxes))
"""
plt.hist(n_boxes)
plt.title('Number of boxes in each image')
plt.show()

plt.bar(range(1, 9), n_class)
plt.title('Distribution of classes')
plt.show()

plt.hist(area_list)
plt.title('Distribution of bounding box area of all classes')
plt.show()

fig, ax = plt.subplots()
bp = ax.boxplot(area_class_list)
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
plt.title('Bounding box area of each class')
plt.xlabel('label')
plt.show()
"""

box_position_list = [xmin_list, ymin_list, xmax_list, ymax_list]
fig, ax = plt.subplots()
bp = ax.boxplot(box_position_list)
ax.set_xticklabels(['xmin', 'ymin', 'xmax', 'ymax'])
plt.title('Bounding box position')
plt.show()