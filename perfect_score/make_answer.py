import numpy as np
import pandas as pd
import os
import xml.dom.minidom

path = './matching.txt'

with open(path) as f:
  l = f.readlines()

xml_list= []

dom = xml.dom.minidom.Document()
root = dom.createElement('annotations')
dom.appendChild(root)

class_convert = {'D00':'1', 'D01':'2', 'D10':'3', 'D11':'4', 'D20':'5', 'D40':'6', 'D43':'7', 'D44':'8'}


for pair in l:
  test, dataset = pair.strip().split(', ')
  location = dataset.split('_')[0]
  file_name = os.path.splitext(dataset)[0]
  answer_xml_path = './RoadDamageDataset/%s/Annotations/%s.xml'%(location, file_name)
  doc = xml.dom.minidom.parse(answer_xml_path)
  annotation = doc.getElementsByTagName('annotation')[0]
  filename = annotation.getElementsByTagName('filename')
  filename[0].childNodes[0].nodeValue = '%s'%(test)
  objects = annotation.getElementsByTagName('object')
  for obj in objects:
    name = obj.getElementsByTagName('name')
    name[0].childNodes[0].nodeValue = class_convert[name[0].childNodes[0].nodeValue]
  root.appendChild(annotation)

xml = dom.toprettyxml()

f = open('perfect.xml', 'w')
f.write(xml)
f.close
