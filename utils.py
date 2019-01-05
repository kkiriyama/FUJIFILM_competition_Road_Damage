import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from tqdm import tqdm

for i in range(1, 8):
  PATH = './keras-yolo3_modified/train/location%d/images'%(i)
  files = os.listdir(PATH)
  for f in tqdm(files):
    img = Image.open(os.path.join(PATH, f))
    img_resize = img.resize((384, 384))
    img_resize.save(f)
