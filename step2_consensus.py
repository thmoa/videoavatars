#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import argparse
import numpy as np
import chumpy as ch
import cPickle as pkl

from opendr.camera import ProjectPoints
from opendr.renderer import BoundaryRenderer, ColoredRenderer
from tqdm import tqdm

from util import im, mesh
from util.logger import log
from lib.frame import setup_frame_rays
from lib.rays import ray_objective
from lib.geometry import laplacian
from lib.ch import sp_dot
from models.smpl import Smpl
from models.bodyparts import faces_no_hands, regularize_laplace, regularize_model, regularize_symmetry


def get_cb(frame, base_smpl, camera, frustum):
    viz_mask = frame.mask / 255.
    base_smpl.pose[:] = frame.pose
    camera.t[:] = frame.trans
    camera.rt[:] = 0

    rn = ColoredRenderer(camera=camera, v=base_smpl, f=base_smpl.f, vc=np.ones_like(base_smpl),
                         frustum=frustum, bgcolor=0, num_channels=1)

    def cb(_):
        silh_diff = (rn.r - viz_mask + 1) / 2.
        im.show(silh_diff, waittime=1)

    return cb


def fit_consensus(frames, base_smpl, camera, frustum, model_data, nohands, icp_count, naked, display):
    if nohands:
        faces = faces_no_hands(base_smpl.f)
    else:
        faces = base_smpl.f

    vis_rn_b = BoundaryRenderer(camera=camera, frustum=frustum, f=faces, num_channels=1)
    vis_rn_m = ColoredRenderer(camera=camera, frustum=frustum, f=faces, vc=np.zeros_like(base_smpl), bgcolor=1,
                               num_channels=1)

    model_template = Smpl(model_data)
    model_template.betas[:] = base_smpl.betas.r

    g_laplace = regularize_laplace()
    g_model = regularize_model()
    g_symmetry = regularize_symmetry()

    for step, (w_laplace, w_model, w_symmetry, sigma) in enumerate(zip(
            np.linspace(6.5, 4.0, icp_count) if naked else np.linspace(4.0, 2.0, icp_count),
            np.linspace(0.9, 0.6, icp_count) if naked else np.linspace(0.6, 0.3, icp_count),
            np.linspace(3.6, 1.8, icp_count),
            np.linspace(0.06, 0.003, icp_count),
    )):
        log.info('# Step {}'.format(step))

        L = laplacian(model_template.r, base_smpl.f)
        delta = L.dot(model_template.r)

        w_laplace *= g_laplace.reshape(-1, 1)
        w_model *= g_model.reshape(-1, 1)
        w_symmetry *= g_symmetry.reshape(-1, 1)

        E = {
            'laplace': (sp_dot(L, base_smpl.v_shaped_personal) - delta) * w_laplace,
            'model': (base_smpl.v_shaped_personal - model_template) * w_model,
            'symmetry': (base_smpl.v_personal + np.array([1, -1, -1])
                         * base_smpl.v_personal[model_data['vert_sym_idxs']]) * w_symmetry,
        }

        log.info('## Matching rays with contours')
        for current, f in enumerate(tqdm(frames)):
            E['silh_{}'.format(current)] = ray_objective(f, sigma, base_smpl, camera, vis_rn_b, vis_rn_m)

        log.info('## Run optimization')
        ch.minimize(
            E,
            [base_smpl.v_personal, model_template.betas],
            method='dogleg',
            options={'maxiter': 15, 'e_3': 0.001},
            callback=get_cb(frames[0], base_smpl, camera, frustum) if display else None
        )


def main(pose_file, masks_file, camera_file, out, obj_out, num, icp_count, model_file, first_frame, last_frame,
         nohands, naked, display):

    # load data
    with open(model_file, 'rb') as fp:
        model_data = pkl.load(fp)

    with open(camera_file, 'rb') as fp:
        camera_data = pkl.load(fp)

    pose_data = h5py.File(pose_file, 'r')
    poses = pose_data['pose'][first_frame:last_frame]
    trans = pose_data['trans'][first_frame:last_frame]
    masks = h5py.File(masks_file, 'r')['masks'][first_frame:last_frame]
    num_frames = masks.shape[0]

    indices_consensus = np.ceil(np.arange(num) * num_frames * 1. / num).astype(np.int)

    # init
    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = np.array(pose_data['betas'], dtype=np.float32)

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)
    camera_t = camera_data['camera_t']
    camera_rt = camera_data['camera_rt']
    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}
    frames = []

    for i in indices_consensus:
        log.info('Set up frame {}...'.format(i))

        mask = np.array(masks[i] * 255, dtype=np.uint8)
        pose_i = np.array(poses[i], dtype=np.float32)
        trans_i = np.array(trans[i], dtype=np.float32)

        frames.append(setup_frame_rays(base_smpl, camera, camera_t, camera_rt, pose_i, trans_i, mask))

    log.info('Set up complete.')
    log.info('Begin consensus fit...')
    fit_consensus(frames, base_smpl, camera, frustum, model_data, nohands, icp_count, naked, display)

    with open(out, 'wb') as fp:
        pkl.dump({
            'v_personal': base_smpl.v_personal.r,
            'betas': base_smpl.betas.r,
        }, fp, protocol=2)

    if obj_out is not None:
        base_smpl.pose[:] = 0
        vt = np.load('assets/basicModel_vt.npy')
        ft = np.load('assets/basicModel_ft.npy')
        mesh.write(obj_out, base_smpl.r, base_smpl.f, vt=vt, ft=ft)

    log.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pose_file',
        type=str,
        help="File that contains poses")
    parser.add_argument(
        'masks_file',
        type=str,
        help="File that contains segmentations")
    parser.add_argument(
        'camera',
        type=str,
        help="pkl file that contains camera settings")
    parser.add_argument(
        'out',
        type=str,
        help="Out file path")
    parser.add_argument(
        '--obj_out', '-oo',
        default=None,
        help='obj out file name (optional)')
    parser.add_argument(
        '--num', '-n', default=120, type=int,
        help="Number of used frames")
    parser.add_argument(
        '--icp', '-i', default=3, type=int,
        help="ICP Iterations")
    parser.add_argument(
        '--model', '-m',
        default='vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        help='Path to SMPL model')
    parser.add_argument(
        '--first_frame', '-f', default=0, type=int,
        help="First frame to use")
    parser.add_argument(
        '--last_frame', '-l', default=2000, type=int,
        help="Last frame to use")
    parser.add_argument(
        '--nohands', '-nh',
        action='store_true',
        help="Exclude hands from optimization")
    parser.add_argument(
        '--naked', '-nk',
        action='store_true',
        help="Person wears (almost) no clothing")
    parser.add_argument(
        '--display', '-d',
        action='store_true',
        help="Enable visualization")

    args = parser.parse_args()

    main(args.pose_file, args.masks_file, args.camera, args.out, args.obj_out, args.num, args.icp, args.model,
         args.first_frame, args.last_frame, args.nohands, args.naked, args.display)
