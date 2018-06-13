#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

from render.renderer import OrthoTexturedRenderer, OrthoColoredRenderer
from render.camera import OrthoProjectPoints


class Isomapper():
    def __init__(self, vt, ft, f, tex_res, bgcolor=np.zeros(3)):
        vt3d = np.dstack((vt[:, 0] - 0.5, 1 - vt[:, 1] - 0.5, np.zeros(vt.shape[0])))[0]
        ortho = OrthoProjectPoints(rt=np.zeros(3), t=np.zeros(3), near=-1, far=1, left=-0.5, right=0.5, bottom=-0.5,
                                   top=0.5, width=tex_res, height=tex_res)
        self.tex_res = tex_res
        self.f = ft
        self.ft = f
        self.rn_tex = OrthoTexturedRenderer(v=vt3d, f=ft, ortho=ortho, vc=np.ones_like(vt3d), bgcolor=bgcolor)
        self.rn_vis = OrthoColoredRenderer(v=vt3d, f=ft, ortho=ortho, vc=np.ones_like(vt3d), bgcolor=np.zeros(3),
                                           num_channels=1)
        self.bgcolor = bgcolor
        self.iso_mask = np.array(self.rn_vis.r)

    def render(self, frame, proj_v, visible_faces=None):
        h, w, _ = np.atleast_3d(frame).shape
        v2d = proj_v.r
        v2d_as_vt = np.dstack((v2d[:, 0] / w, 1 - v2d[:, 1] / h))[0]

        self.rn_tex.set(texture_image=frame, vt=v2d_as_vt, ft=self.ft)
        tex = np.array(self.rn_tex.r)

        if visible_faces is not None:
            self.rn_vis.set(f=self.f[visible_faces])
            mask = np.atleast_3d(self.rn_vis.r)
            tex = mask * tex + (1 - mask) * self.bgcolor

        return tex


class IsoColoredRenderer:
    def __init__(self, vt, ft, f, tex_res):
        ortho = OrthoProjectPoints(rt=np.zeros(3), t=np.zeros(3), near=-1, far=1, left=-0.5, right=0.5, bottom=-0.5,
                                   top=0.5, width=tex_res, height=tex_res)
        vt3d = np.dstack((vt[:, 0] - 0.5, 1 - vt[:, 1] - 0.5, np.zeros(vt.shape[0])))[0]
        vt3d = vt3d[ft].reshape(-1, 3)
        self.f = f
        self.rn = OrthoColoredRenderer(bgcolor=np.zeros(3), ortho=ortho, v=vt3d, f=np.arange(ft.size).reshape(-1, 3))

    def render(self, vc):
        vc = np.atleast_3d(vc)

        if vc.shape[2] == 1:
            vc = np.hstack((vc, vc, vc))

        self.rn.set(vc=vc[self.f].reshape(-1, 3))

        return np.array(self.rn.r)
