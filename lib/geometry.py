#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import sparse as sp


def visible_boundary_edges(rn_b, rn_m):
    visibility = rn_b.boundaryid_image

    silh = rn_m.r
    sobelx = cv2.Sobel(silh, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(silh, cv2.CV_64F, 0, 1, ksize=3)

    mag = (sobelx ** 2 + sobely ** 2) > 0

    visibility[mag == 0] = 4294967295
    visible = np.nonzero(visibility.ravel() != 4294967295)[0]

    return np.unique(visibility.ravel()[visible])


def visible_boundary_edge_verts(rn_b, rn_m):
    visible_edge_ids = visible_boundary_edges(rn_b, rn_m)

    vpe = rn_b.primitives_per_edge[1]
    verts = np.unique(vpe[visible_edge_ids].ravel())

    return verts


def laplacian(v, f):
    n = len(v)

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    ab = v[v_a] - v[v_b]
    bc = v[v_b] - v[v_c]
    ca = v[v_c] - v[v_a]

    cot_a = -1 * (ab * ca).sum(axis=1) / np.sqrt(np.sum(np.cross(ab, ca) ** 2, axis=-1))
    cot_b = -1 * (bc * ab).sum(axis=1) / np.sqrt(np.sum(np.cross(bc, ab) ** 2, axis=-1))
    cot_c = -1 * (ca * bc).sum(axis=1) / np.sqrt(np.sum(np.cross(ca, bc) ** 2, axis=-1))

    I = np.concatenate((v_a, v_c, v_a, v_b, v_b, v_c))
    J = np.concatenate((v_c, v_a, v_b, v_a, v_c, v_b))
    W = 0.5 * np.concatenate((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a))

    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    L = L - sp.spdiags(L * np.ones(n), 0, n, n)

    return L
