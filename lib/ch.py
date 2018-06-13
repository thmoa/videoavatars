#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import chumpy as ch
import scipy.sparse as sp

from chumpy.utils import col


class sp_dot(ch.Ch):
    terms = 'a',
    dterms = 'b',

    def compute_r(self):
        return self.a.dot(self.b.r)

    def compute(self):

        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        ar = sp.csr_matrix((self.a.data, self.a.indices, self.a.indptr),
                            shape=(max(np.sum(self.a.shape[:-1]), 1), self.a.shape[-1]))
        br = col(self.b.r) if len(self.b.r.shape) < 2 else self.b.r.reshape((self.b.r.shape[0], -1))

        if br.ndim <= 1:
            return ar
        elif br.ndim <= 2:
            return sp.kron(ar, sp.eye(br.shape[1], br.shape[1]))
        else:
            raise NotImplementedError

    def compute_dr_wrt(self, wrt):
        if wrt is self.b:
            return self.compute()