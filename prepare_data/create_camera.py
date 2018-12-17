#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cPickle as pkl

"""

This script creates a .pkl file using the given camera intrinsics.

Example:
$ python create_camera.py camera.pkl 1080 1080 -f 900.0 900.0

"""

parser = argparse.ArgumentParser()
parser.add_argument('out', type=str, help="Output file (.pkl)")
parser.add_argument('width', type=int, help="Frame width in px")
parser.add_argument('height', type=int, help="Frame height in px")
parser.add_argument('-f', type=float, nargs='*', help="Focal length in px (2,)")
parser.add_argument('-c', type=float, nargs='*', help="Principal point in px (2,)")
parser.add_argument('-k', type=float, nargs='*', help="Distortion coefficients (5,)")

args = parser.parse_args()

camera_data = {
    'camera_t': np.zeros(3),
    'camera_rt': np.zeros(3),
    'camera_f': np.array([args.width, args.width]),
    'camera_c': np.array([args.width, args.height]) / 2.,
    'camera_k': np.zeros(5),
    'width': args.width,
    'height': args.height,
}

if args.f is not None:
    if len(args.f) is not 2:
        raise Exception('Focal length should be of shape (2,)')

    camera_data['camera_f'] = np.array(args.f)

if args.c is not None:
    if len(args.c) is not 2:
        raise Exception('Principal point should be of shape (2,)')

    camera_data['camera_c'] = np.array(args.c)

if args.k is not None:
    if len(args.k) is not 5:
        raise Exception('Distortion coefficients should be of shape (5,)')

    camera_data['camera_k'] = np.array(args.k)

with open(args.out, 'wb') as f:
    pkl.dump(camera_data, f, protocol=2)
