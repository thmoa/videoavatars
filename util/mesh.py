#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np


def write(filename, v, f, vt=None, ft=None, vn=None, vc=None, texture=None):
    with open(filename, 'w') as fp:
        if texture is not None:
            mat_file = filename.replace('obj', 'mtl')

            fp.write('mtllib {}\n'.format(os.path.basename(mat_file)))
            fp.write('usemtl mat\n')

            with open(mat_file, 'w') as mfp:
                mfp.write('newmtl mat\n')
                mfp.write('Ka 1.0 1.0 1.0\n')
                mfp.write('Kd 1.0 1.0 1.0\n')
                mfp.write('Ks 0.0 0.0 0.0\n')
                mfp.write('d 1.0\n')
                mfp.write('Ns 0.0\n')
                mfp.write('illum 0\n')
                mfp.write('map_Kd {}\n'.format(texture))

        if vc is not None:
            fp.write(('v {:f} {:f} {:f} {:f} {:f} {:f}\n' * len(v)).format(*np.hstack((v, vc)).reshape(-1)))
        else:
            fp.write(('v {:f} {:f} {:f}\n' * len(v)).format(*v.reshape(-1)))

        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * len(vn)).format(*vn.reshape(-1)))

        if vt is not None:
            fp.write(('vt {:f} {:f}\n' * len(vt)).format(*vt.reshape(-1)))

        if ft is not None:
            fp.write(('f {:d}/{:d}/{:d} {:d}/{:d}/{:d} {:d}/{:d}/{:d}\n' * len(f)).format(*np.hstack((f.reshape(-1, 1), ft.reshape(-1, 1), f.reshape(-1, 1))).reshape(-1) + 1))
        else:
            fp.write(('f {:d}//{:d} {:d}//{:d} {:d}//{:d}\n' * len(f)).format(*np.repeat(f.reshape(-1) + 1, 2)))
