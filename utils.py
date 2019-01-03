import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from tqdm import tqdm

for i in range(1, 8):
  PATH = '../content/drive/My Drive/fujifilm/detection/keras-0ylo3_modified/train/location%d/images'%(i)
  files = os.listdir(PATH)
  print(len(files))
  for f in tqdm(files):
    img = Image.open(os.path.join(PATH + f))
    img_resize = img.resize((416, 416))
    img_resize.save(f)
