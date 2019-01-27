import numpy as np
import pandas as pd
import os
import glob
import logging
import sys
from tqdm import tqdm
from PIL import Image

location = ["Adachi", "Chiba", "Ichihara", "Muroran", "Nagakute", "Numazu", "Sumida"]

filenames = os.listdir('./test')

path_w = './matching.txt'
img_size = (5, 5)
imgs = []

for i in range(7):
  sub_imgs = []
  comparing_dir_path = './RoadDamageDataset/%s/JPEGImages'%(location[i])
  comparing_files = os.listdir(comparing_dir_path)
  for comparing_file in tqdm(comparing_files):
    comparing_img_path = os.path.join(comparing_dir_path, comparing_file)
    comparing_img = Image.open(comparing_img_path)
    comparing_img = comparing_img.resize(img_size)
    comparing_img = np.array(comparing_img)
    comparing_img = comparing_img.flatten()
    sub_imgs.append(comparing_img)
  imgs.append(sub_imgs)

for filename in filenames:

  target_file_path = './test/%s'%(filename)

  loc_ret = {}

  for i in range(7):
    ret = {}

    comparing_dir_path = './RoadDamageDataset/%s/JPEGImages'%(location[i])
    pattern = '%s/*.jpg'
    comparing_files = os.listdir(comparing_dir_path)
    target_file_name = os.path.basename(target_file_path)
    target_img = Image.open(target_file_path)
    target_img = target_img.resize(img_size)
    target_img = np.array(target_img)
    target_img = target_img.flatten()

    for k, comparing_file in tqdm(enumerate(comparing_files)):
      comparing_file_name = os.path.basename(comparing_file)
      
      comparing_img = imgs[i][k]

      # detect
      corr = np.corrcoef([target_img, comparing_img])
      ret[comparing_file] = corr[0][1]

    min_value = max(ret.values())
    min_file_name = [k for k, v in ret.items() if v == min_value][0]

    loc_ret[min_file_name] = min_value
    print(min_value, min_file_name)

  min_min_value = max(loc_ret.values())
  min_min_file_name = [k for k, v in loc_ret.items() if v == min_min_value][0]

  with open(path_w, mode = 'a') as f:
    f.write('%s, %s\n'%(filename, min_min_file_name))
  






