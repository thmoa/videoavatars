#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import platform
import numpy as np

from chumpy import Ch, depends_on
from opendr.renderer import BaseRenderer, ColoredRenderer, TexturedRenderer
from opendr.renderer import draw_edge_visibility, draw_boundary_images, draw_boundaryid_image

if platform.system() == 'Darwin':
    from opendr.contexts.ctx_mac import OsContext
else:
    from opendr.contexts.ctx_mesa import OsContext
from opendr.contexts._constants import *


class OrthoBaseRenderer(BaseRenderer):
    terms = ['f', 'overdraw']
    dterms = ['ortho', 'v']

    @property
    def v(self):
        return self.ortho.v

    @v.setter
    def v(self, newval):
        self.ortho.v = newval

    @depends_on('f', 'ortho', 'overdraw')
    def barycentric_image(self):
        return super(OrthoBaseRenderer, self).barycentric_image

    @depends_on(terms+dterms)
    def boundaryid_image(self):
        self._call_on_changed()
        return draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.ortho)

    @depends_on('f', 'ortho', 'overdraw')
    def visibility_image(self):
        return super(OrthoBaseRenderer, self).visibility_image

    @depends_on('f', 'ortho')
    def edge_visibility_image(self):
        self._call_on_changed()
        return draw_edge_visibility(self.glb, self.v.r, self.vpe, self.f)


class OrthoColoredRenderer(OrthoBaseRenderer, ColoredRenderer):
    terms = 'f', 'background_image', 'overdraw', 'num_channels'
    dterms = 'vc', 'ortho', 'bgcolor'

    def compute_r(self):
        return self.color_image

    def compute_dr_wrt(self, wrt):
        raise NotImplementedError

    def on_changed(self, which):
        if 'ortho' in which:
            w = self.ortho.width
            h = self.ortho.height
            self.glf = OsContext(np.int(w), np.int(h), typ=GL_FLOAT)
            _setup_ortho(self.glf, self.ortho.left.r, self.ortho.right.r, self.ortho.bottom.r, self.ortho.top.r,
                         self.ortho.near, self.ortho.far, self.ortho.view_mtx)
            self.glf.Viewport(0, 0, w, h)
            self.glb = OsContext(np.int(w), np.int(h), typ=GL_UNSIGNED_BYTE)
            self.glb.Viewport(0, 0, w, h)
            _setup_ortho(self.glb, self.ortho.left.r, self.ortho.right.r, self.ortho.bottom.r, self.ortho.top.r,
                         self.ortho.near, self.ortho.far, self.ortho.view_mtx)

        if not hasattr(self, 'num_channels'):
            self.num_channels = 3

        if not hasattr(self, 'bgcolor'):
            self.bgcolor = Ch(np.array([.5] * self.num_channels))
            which.add('bgcolor')

        if not hasattr(self, 'overdraw'):
            self.overdraw = True

        if 'bgcolor' in which:
            self.glf.ClearColor(self.bgcolor.r[0], self.bgcolor.r[1 % self.num_channels],
                                self.bgcolor.r[2 % self.num_channels], 1.)

    @depends_on('f', 'ortho', 'vc')
    def boundarycolor_image(self):
        return self.draw_boundarycolor_image(with_vertex_colors=True)

    @depends_on('f', 'ortho')
    def boundary_images(self):
        self._call_on_changed()
        return draw_boundary_images(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.ortho)

    @depends_on(terms+dterms)
    def color_image(self):
        return super(OrthoColoredRenderer, self).color_image

    @property
    def shape(self):
        return (self.ortho.height, self.ortho.width, 3)


class OrthoTexturedRenderer(OrthoColoredRenderer, TexturedRenderer):
    terms = 'f', 'ft', 'background_image', 'overdraw', 'tex_filter_mag', 'tex_filter_min'
    dterms = 'vc', 'ortho', 'bgcolor', 'texture_image', 'vt'

    def compute_dr_wrt(self, wrt):
        raise NotImplementedError

    def on_changed(self, which):
        OrthoColoredRenderer.on_changed(self, which)

        # have to redo if ortho changes, b/c ortho triggers new context
        if 'texture_image' in which or 'ortho' in which:
            gl = self.glf
            texture_data = np.array(self.texture_image * 255., dtype='uint8', order='C')
            tmp = np.zeros(1, dtype=np.uint32)

            self.release_textures()
            gl.GenTextures(1, tmp)

            self.textureID = tmp[0]
            gl.BindTexture(GL_TEXTURE_2D, self.textureID)

            gl.TexImage2Dub(GL_TEXTURE_2D, 0, GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL_RGB,
                            texture_data.ravel())
            gl.GenerateMipmap(GL_TEXTURE_2D)

    def release_textures(self):
        if hasattr(self, 'textureID'):
            arr = np.asarray(np.array([self.textureID]), np.uint32)
            self.glf.DeleteTextures(arr)

    def texture_mapping_on(self, gl, with_vertex_colors):
        gl.Enable(GL_TEXTURE_2D)
        gl.BindTexture(GL_TEXTURE_2D, self.textureID)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        gl.TexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE if with_vertex_colors else GL_REPLACE)
        gl.EnableClientState(GL_TEXTURE_COORD_ARRAY)

    @depends_on(dterms+terms)
    def boundaryid_image(self):
        return super(OrthoTexturedRenderer, self).boundaryid_image

    @depends_on(terms+dterms)
    def color_image(self):
        self.glf.BindTexture(GL_TEXTURE_2D, self.textureID)
        return super(OrthoTexturedRenderer, self).color_image

    @depends_on(terms+dterms)
    def boundarycolor_image(self):
        self.glf.BindTexture(GL_TEXTURE_2D, self.textureID)
        return super(OrthoTexturedRenderer, self).boundarycolor_image

    @property
    def shape(self):
        return (self.ortho.height, self.ortho.width, 3)

    @depends_on('vt', 'ft')
    def mesh_tex_coords(self):
        ftidxs = self.ft.ravel()
        data = np.asarray(self.vt.r[ftidxs].astype(np.float32)[:, 0:2], np.float32, order='C')
        data[:, 1] = 1.0 - 1.0 * data[:, 1]
        return data


def _setup_ortho(gl, l, r, b, t, near, far, view_matrix):
    gl.MatrixMode(GL_PROJECTION)
    gl.LoadIdentity()
    gl.Ortho(l, r, t, b, near, far)  # top and bottom switched for opencv coordinate system

    gl.MatrixMode(GL_MODELVIEW)
    gl.LoadIdentity()
    gl.Rotatef(180, 1, 0, 0)

    view_mtx = np.asarray(np.vstack((view_matrix, np.array([0, 0, 0, 1]))), np.float32, order='F')
    gl.MultMatrixf(view_mtx)

    gl.Enable(GL_DEPTH_TEST)
    gl.PolygonMode(GL_BACK, GL_FILL)
    gl.Disable(GL_LIGHTING)
    gl.Disable(GL_CULL_FACE)
    gl.PixelStorei(GL_PACK_ALIGNMENT, 1)
    gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1)

    gl.UseProgram(0)
