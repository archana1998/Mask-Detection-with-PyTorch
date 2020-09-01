import argparse
import json
import torch 
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

classes = ['with_mask', 'without_mask']
parser = argparse.ArgumentParser()
parser.add_argument('-rootdir', type=str, default='root', help='Directory of training data')
args = parser.parse_args()

root_dir = Path(args.rootdir)
mask_path = root_dir/'with_mask'
nonmask_path = root_dir/'without_mask'
print(nonmask_path)
maskdata_df = pd.DataFrame()

if not os.path.isfile('data/mask_df.pickle'):
    for subject in tqdm(list(nonmask_path.iterdir()), desc='non mask photos'):
        image = cv2.imread(str(subject))
        maskdata_df = maskdata_df.append({
        'image': image,
        'mask': 0
    }, ignore_index=True)
    for subject in tqdm(list(mask_path.iterdir()), desc='mask photos'):
        image = cv2.imread(str(subject))
        maskdata_df = maskdata_df.append({
            'image': image,
            'mask': 1
    }, ignore_index=True)
    pickle_path = Path('data')
    if not os.path.isdir(pickle_path):
        os.mkdir(pickle_path)
    maskdata_df.to_pickle('data/mask_df.pickle')

else: 
    print('Dataframe exists')

"""
class MaskDetectionDataset(Dataset):
    Mask Detection Dataset
    def __init__ (self, root_dir: str, train: bool = True, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if not self.root_dir.exists():
            raise ValueError('Mask Detection images were not found at location {0}.'.format(str(self.root_dir)))

    
    def __getitem__(self, idx):

 """       