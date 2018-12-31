import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

sets = range(1, 8)

def count_annotation(location_n, image_id):
  image_id = image_id.split('.')[0]
  in_file = open('../train/location%d/labels/%s.xml'%(location_n, image_id))
  tree=ET.parse(in_file)
  root = tree.getroot()

  c = 0
  counter = np.zeros(8)
  for obj in root.iter('object'):
    cls_id = int(obj.find('name').text)
    counter[cls_id - 1] += 1
    c += 1

  return c, counter

n_boxes = []
n_class = np.zeros(8)

for location_n in sets:
  image_ids = os.listdir('../train/location%d/images'%(location_n))
  for image_id in image_ids:
    n_box, n_cls = count_annotation(location_n, image_id)
    n_boxes.append(n_box)
    n_class += n_cls

print(max(n_boxes), min(n_boxes))

plt.hist(n_boxes)
plt.title('Number of boxes in each image')
plt.show()

plt.bar(range(1, 9), n_class)
plt.title('Distribution of classes')
plt.show()    