#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import h5py
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

"""

This script stores image masks from a directory in a compressed hdf5 file.

Example:
$ python masks2hdf5.py dataset/subject/masks masks.hdf5

"""

parser = argparse.ArgumentParser()
parser.add_argument('src_folder', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
mask_dir = args.src
mask_files = sorted(glob(os.path.join(mask_dir, '*.png')) + glob(os.path.join(mask_dir, '*.jpg')))

with h5py.File(out_file, 'w') as f:
    dset = None

    for i, silh_file in enumerate(tqdm(mask_files)):
        silh = cv2.imread(silh_file, cv2.IMREAD_GRAYSCALE)

        if dset is None:
            dset = f.create_dataset("masks", (len(mask_files), silh.shape[0], silh.shape[1]), 'b', chunks=True, compression="lzf")

        _, silh = cv2.threshold(silh, 100, 255, cv2.THRESH_BINARY)
        dset[i] = silh.astype(np.bool)
