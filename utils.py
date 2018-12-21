import numpy as np
import pandas as pd
import os
import glob
from PIL import Image

for i in range(1, 8):
  files = glob.glob('./train/location%s/images/*.jpg'%(i))
  for f in files:
    img = Image.open(f)
    img_resize = img.resize((192, 192))
    img_resize.save(f)
