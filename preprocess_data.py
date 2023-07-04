#%%
from PIL import Image
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from matplotlib import pyplot as plt
from os.path import exists, join, dirname
from pathlib import Path

import os
import shutil
from tqdm import tqdm
from datetime import datetime
from glob import glob

TARGET_MODAL = 'radar'    # radar | thermal

SRC_DIR = 'datasets/fusion'
DST_DIR = f'datasets/{TARGET_MODAL}'
for SPLIT in ['test']:
    src_files = glob(join(SRC_DIR, SPLIT, 'images/*.png'))
    for src_file in tqdm(src_files):
        path, name = os.path.split(src_file)
        img = Image.open(src_file)
        img_np = np.asarray(img) # (H,W,C)

        if TARGET_MODAL == 'radar':
            img_np[...,1] = img_np[...,0] # radar
            img_np[...,2] = img_np[...,0] # radar
        elif TARGET_MODAL == 'thermal':
            img_np[...,0] = img_np[...,1] # thermal

        img = Image.fromarray(np.uint8(img_np)).convert('RGB')
        dst_file = join(DST_DIR, SPLIT, name)
        img.save(dst_file)

#%%  sanity check
# im1 = np.asarray(Image.open(src_file))
# im2 = np.asarray(Image.open(dst_file))
# (im1 - im2).sum()
