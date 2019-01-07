import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def resize(size = 416):
  for i in range(1, 8):
    PATH = './keras-yolo3_modified/train/location%d/images'%(i)
    files = os.listdir(PATH)
    for f in tqdm(files):
      img = Image.open(os.path.join(PATH, f))
      img_resize = img.resize((size, size))
      img_resize.save(f)

def adaptive_th():
  for i in range(1, 8):
    PATH = './keras-yolo3_modified/train/location%d/images/'%(i)
    SAVE_PATH = './keras-yolo3_modified/train_preprocessed/location%d/images/'%(i)
    train_list = os.listdir(PATH)
    for path in tqdm(train_list):
      img = cv2.imread(PATH + path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 10)
      th = cv2.bilateralFilter(th, 20, 100, 20)
      th = cv2.bilateralFilter(th, 20, 100, 20)
      th = cv2.bilateralFilter(th, 20, 100, 20)
      cv2.imwrite(SAVE_PATH + path, th)
      