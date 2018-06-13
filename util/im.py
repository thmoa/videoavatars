#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


_figures = {}


def show(im, waittime=0, id='plt', max_width=600):
    plt.ion()
    w = min(im.shape[1], max_width)
    h = max_width * (1.0 * im.shape[0]) / im.shape[1] if w == max_width else im.shape[0]
    plt.figure(id, figsize=(w / 80, h / 80), dpi=80)

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if np.issubdtype(im.dtype, np.floating):
        if np.max(im) > 1:
            factor = 255 / np.max(im)
        else:
            factor = 255
    else:
        factor = 1

    if np.atleast_3d(im).shape[2] == 3:
        data = np.uint8(im * factor)[:, :, ::-1]
    else:
        data = np.uint8(np.dstack((im, im, im)) * factor)

    if id in _figures and plt.fignum_exists(id):
        _figures[id].set_array(data)
    else:
        _figures[id] = plt.imshow(data)

    if waittime == 0:
        plt.waitforbuttonpress()
    else:
        plt.pause(waittime / 1000.)
    plt.ioff()
