#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import chumpy as ch
from chumpy import Ch, depends_on
from chumpy.utils import col
from opendr.geometry import Rodrigues


class OrthoProjectPoints(Ch):
    terms = 'near', 'far', 'width', 'height'
    dterms = 'v', 'rt', 't', 'left', 'right', 'bottom', 'top'

    def compute_r(self):
        return self.r_and_derivatives.r

    def compute_dr_wrt(self, wrt):
        if wrt not in [self.v, self.rt, self.t, self.left, self.right, self.bottom, self.top]:
            return None

        return self.r_and_derivatives.dr_wrt(wrt)

    @depends_on('t', 'rt')
    def view_mtx(self):
        R = cv2.Rodrigues(self.rt.r)[0]
        return np.hstack((R, col(self.t.r)))

    @property
    def r_and_derivatives(self):
        tmp = self.v.dot(Rodrigues(self.rt)) + self.t

        return ch.hstack((
            col(2. / (self.right - self.left) * tmp[:, 0] - (self.right + self.left) / (self.right - self.left) + 1.) * self.width / 2.,
            col(2. / (self.bottom - self.top) * tmp[:, 1] - (self.bottom + self.top) / (self.bottom - self.top) + 1.) * self.height / 2.,
        ))

