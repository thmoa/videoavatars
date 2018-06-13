#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

from util.logger import log
from lib.rays import rays_from_silh
from models.smpl import model_params_in_camera_coords

sess = None


class FrameData(object):
    pass


def batch_invert(x):
    try:
        import tensorflow as tf
        global sess

        tx = tf.convert_to_tensor(x, dtype=tf.float32)
        txi = tf.transpose(tf.matrix_inverse(tf.transpose(tx)))

        if sess is None:
            sess = tf.Session()

        return sess.run(txi)

    except ImportError:
        log.info('Could not load tensorflow. Falling back to matrix inversion with numpy (slower).')

    return np.asarray([np.linalg.inv(t) for t in x.T]).T


def setup_frame_rays(base_smpl, camera, camera_t, camera_rt, pose, trans, mask):
    f = FrameData()

    f.trans, f.pose = model_params_in_camera_coords(trans, pose, base_smpl.J[0], camera_t, camera_rt)
    f.mask = mask

    base_smpl.pose[:] = f.pose
    camera.t[:] = f.trans

    f.Vi = batch_invert(base_smpl.V.r)
    f.rays = rays_from_silh(f.mask, camera)

    return f
