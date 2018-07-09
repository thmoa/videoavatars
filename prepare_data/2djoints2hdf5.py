#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import h5py
import json
import numpy as np

from glob import glob
from tqdm import tqdm

"""

This script stores OpenPose 2D keypoints from json files in the given directory in a compressed hdf5 file.

Example:
$ python 2djoints2hdf5.py dataset/subject/openpose_detections keypoints.hdf5

"""


parser = argparse.ArgumentParser()
parser.add_argument('src_folder', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
pose_dir = args.src_folder
pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

with h5py.File(out_file, 'w') as f:
    poses_dset = f.create_dataset("keypoints", (len(pose_files), 54), 'f', chunks=True, compression="lzf")

    for i, pose_file in enumerate(tqdm(pose_files)):
        with open(pose_file) as fp:
            pose = np.array(json.load(fp)['people'][0]['pose_keypoints_2d'])
            poses_dset[i] = pose
