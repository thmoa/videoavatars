#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import argparse
import numpy as np
import chumpy as ch
import cPickle as pkl

from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from opendr.filters import gaussian_pyramid

from util import im
from util.logger import log
from lib.frame import FrameData
from models.smpl import Smpl, copy_smpl, joints_coco
from models.bodyparts import faces_no_hands

from vendor.smplify.sphere_collisions import SphereCollisions
from vendor.smplify.robustifiers import GMOf


def get_cb(viz_rn, f):
    if viz_rn is not None:
        viz_rn.set(v=f.smpl, background_image=np.dstack((f.mask, f.mask, f.mask)))
        viz_rn.vc.set(v=f.smpl)

        def cb(_):
            debug = np.array(viz_rn.r)

            for j in f.J_proj.r:
                cv2.circle(debug, tuple(j.astype(np.int)), 3, (0, 0, 0.8), -1)
            for j in f.keypoints[:, :2]:
                cv2.circle(debug, tuple(j.astype(np.int)), 3, (0, 0.8, 0), -1)

            im.show(debug, id='pose', waittime=1)
    else:
        cb = None

    return cb


def collision_obj(smpl, regs):
    sp = SphereCollisions(pose=smpl.pose, betas=smpl.betas, model=smpl, regs=regs)
    sp.no_hands = True

    return sp


def pose_prior_obj(smpl, prior_data):
    return (smpl.pose[3:] - prior_data['mean']).reshape(1, -1).dot(prior_data['prec'])


def height_predictor(b2m, betas):
    return ch.hstack((betas.reshape(1, -1), [[1]])).dot(b2m)


def init(frames, body_height, b2m, viz_rn):
    betas = frames[0].smpl.betas

    E_height = None
    if body_height is not None:
         E_height = height_predictor(b2m, betas) - body_height * 1000.

    # first get a rough pose for all frames individually
    for i, f in enumerate(frames):
        if np.sum(f.keypoints[[0, 2, 5, 8, 11], 2]) > 3.:
            if f.keypoints[2, 0] > f.keypoints[5, 0]:
                f.smpl.pose[0] = 0
                f.smpl.pose[2] = np.pi

            E_init = {
                'init_pose_{}'.format(i): f.pose_obj[[0, 2, 5, 8, 11]]
            }

            x0 = [f.smpl.trans, f.smpl.pose[:3]]

            if E_height is not None and i == 0:
                E_init['height'] = E_height
                E_init['betas'] = betas
                x0.append(betas)

            ch.minimize(
                E_init,
                x0,
                method='dogleg',
                options={
                    'e_3': .01,
                },
                callback=get_cb(viz_rn, f)
            )

    weights = zip(
        [5., 4.5, 4.],
        [5., 4., 3.]
    )

    E_betas = betas - betas.r

    for w_prior, w_betas in weights:
        x0 = [betas]

        E = {
            'betas': E_betas * w_betas,
        }

        if E_height is not None:
            E['height'] = E_height

        for i, f in enumerate(frames):
            if np.sum(f.keypoints[[0, 2, 5, 8, 11], 2]) > 3.:
                x0.extend([f.smpl.pose[range(21) + range(27, 30) + range(36, 60)], f.smpl.trans])
                E['pose_{}'.format(i)] = f.pose_obj
                E['prior_{}'.format(i)] = f.pose_prior_obj * w_prior

        ch.minimize(
            E,
            x0,
            method='dogleg',
            options={
                'e_3': .01,
            },
            callback=get_cb(viz_rn, frames[0])
        )


def reinit_frame(frame, null_pose, nohands, viz_rn):

    if (np.sum(frame.pose_obj.r ** 2) > 625 or np.sum(frame.pose_prior_obj.r ** 2) > 75)\
            and np.sum(frame.keypoints[[0, 2, 5, 8, 11], 2]) > 3.:

        log.info('Tracking error too large. Re-init frame...')

        x0 = [frame.smpl.pose[:3], frame.smpl.trans]

        frame.smpl.pose[3:] = null_pose
        if frame.keypoints[2, 0] > frame.keypoints[5, 0]:
            frame.smpl.pose[0] = 0
            frame.smpl.pose[2] = np.pi

        E = {
            'init_pose': frame.pose_obj[[0, 2, 5, 8, 11]],
        }

        ch.minimize(
            E,
            x0,
            method='dogleg',
            options={
                'e_3': .1,
            },
            callback=get_cb(viz_rn, frame)
        )

        E = {
            'pose': GMOf(frame.pose_obj, 100),
            'prior': frame.pose_prior_obj * 8.,
        }

        x0 = [frame.smpl.trans]

        if nohands:
            x0.append(frame.smpl.pose[range(21) + range(27, 30) + range(36, 60)])
        else:
            x0.append(frame.smpl.pose[range(21) + range(27, 30) + range(36, 72)])

        ch.minimize(
            E,
            x0,
            method='dogleg',
            options={
                'e_3': .01,
            },
            callback=get_cb(viz_rn, frame)
        )


def fit_pose(frame, last_smpl, frustum, nohands, viz_rn):

    if nohands:
        faces = faces_no_hands(frame.smpl.f)
    else:
        faces = frame.smpl.f

    dst_type = cv2.cv.CV_DIST_L2 if cv2.__version__[0] == '2' else cv2.DIST_L2

    dist_i = cv2.distanceTransform(np.uint8(frame.mask * 255), dst_type, 5) - 1
    dist_i[dist_i < 0] = 0
    dist_i[dist_i > 50] = 50
    dist_o = cv2.distanceTransform(255 - np.uint8(frame.mask * 255), dst_type, 5)
    dist_o[dist_o > 50] = 50

    rn_m = ColoredRenderer(camera=frame.camera, v=frame.smpl, f=faces, vc=np.ones_like(frame.smpl), frustum=frustum,
                           bgcolor=0, num_channels=1)

    E = {
        'mask': gaussian_pyramid(rn_m * dist_o * 100. + (1 - rn_m) * dist_i, n_levels=4, normalization='size') * 80.,
        '2dpose': GMOf(frame.pose_obj, 100),
        'prior': frame.pose_prior_obj * 4.,
        'sp': frame.collision_obj * 1e3,
    }

    if last_smpl is not None:
        E['last_pose'] = GMOf(frame.smpl.pose - last_smpl.pose, 0.05) * 50.
        E['last_trans'] = GMOf(frame.smpl.trans - last_smpl.trans, 0.05) * 50.

    if nohands:
        x0 = [frame.smpl.pose[range(21) + range(27, 30) + range(36, 60)], frame.smpl.trans]
    else:
        x0 = [frame.smpl.pose[range(21) + range(27, 30) + range(36, 72)], frame.smpl.trans]

    ch.minimize(
        E,
        x0,
        method='dogleg',
        options={
            'e_3': .01,
        },
        callback=get_cb(viz_rn, frame)
    )


def main(keypoint_file, masks_file, camera_file, out, model_file, prior_file, resize,
         body_height, nohands, display):

    # load data
    with open(model_file, 'rb') as fp:
        model_data = pkl.load(fp)

    with open(camera_file, 'rb') as fp:
        camera_data = pkl.load(fp)

    with open(prior_file, 'rb') as fp:
        prior_data = pkl.load(fp)

    if 'basicModel_f' in model_file:
        regs = np.load('vendor/smplify/models/regressors_locked_normalized_female.npz')
        b2m = np.load('assets/b2m_f.npy')
    else:
        regs = np.load('vendor/smplify/models/regressors_locked_normalized_male.npz')
        b2m = np.load('assets/b2m_m.npy')

    keypoints = h5py.File(keypoint_file, 'r')['keypoints']
    masks = h5py.File(masks_file, 'r')['masks']
    num_frames = masks.shape[0]

    # init
    base_smpl = Smpl(model_data)
    base_smpl.trans[:] = np.array([0, 0, 3])
    base_smpl.pose[0] = np.pi
    base_smpl.pose[3:] = prior_data['mean']

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'] * resize,
                           f=camera_data['camera_f'] * resize, k=camera_data['camera_k'], v=base_smpl)
    frustum = {'near': 0.1, 'far': 1000.,
               'width': int(camera_data['width'] * resize), 'height': int(camera_data['height'] * resize)}

    if display:
        debug_cam = ProjectPoints(v=base_smpl, t=camera.t, rt=camera.rt, c=camera.c, f=camera.f, k=camera.k)
        debug_light = LambertianPointLight(f=base_smpl.f, v=base_smpl, num_verts=len(base_smpl), light_pos=np.zeros(3),
                                           vc=np.ones(3), light_color=np.ones(3))
        debug_rn = ColoredRenderer(camera=debug_cam, v=base_smpl, f=base_smpl.f, vc=debug_light, frustum=frustum)
    else:
        debug_rn = None

    # generic frame loading function
    def create_frame(i, smpl, copy=True):
        f = FrameData()

        f.smpl = copy_smpl(smpl, model_data) if copy else smpl
        f.camera = ProjectPoints(v=f.smpl, t=camera.t, rt=camera.rt, c=camera.c, f=camera.f, k=camera.k)

        f.keypoints = np.array(keypoints[i]).reshape(-1, 3) * np.array([resize, resize, 1])
        f.J = joints_coco(f.smpl)
        f.J_proj = ProjectPoints(v=f.J, t=camera.t, rt=camera.rt, c=camera.c, f=camera.f, k=camera.k)
        f.mask = cv2.resize(np.array(masks[i], dtype=np.float32), (0, 0),
                            fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)

        f.collision_obj = collision_obj(f.smpl, regs)
        f.pose_prior_obj = pose_prior_obj(f.smpl, prior_data)
        f.pose_obj = (f.J_proj - f.keypoints[:, :2]) * f.keypoints[:, 2].reshape(-1, 1)

        return f

    base_frame = create_frame(0, base_smpl, copy=False)

    # get betas from 5 frames
    log.info('Initial fit')

    num_init = 5
    indices_init = np.ceil(np.arange(num_init) * num_frames * 1. / num_init).astype(np.int)

    init_frames = [base_frame]
    for i in indices_init[1:]:
        init_frames.append(create_frame(i, base_smpl))

    init(init_frames, body_height, b2m, debug_rn)

    # get pose frame by frame
    with h5py.File(out, 'w') as fp:
        last_smpl = None
        poses_dset = fp.create_dataset("pose", (num_frames, 72), 'f', chunks=True, compression="lzf")
        trans_dset = fp.create_dataset("trans", (num_frames, 3), 'f', chunks=True, compression="lzf")
        betas_dset = fp.create_dataset("betas", (10,), 'f', chunks=True, compression="lzf")

        for i in xrange(num_frames):
            if i == 0:
                current_frame = base_frame
            else:
                current_frame = create_frame(i, last_smpl)

            log.info('Fit frame {}'.format(i))
            # re-init if necessary
            reinit_frame(current_frame, prior_data['mean'], nohands, debug_rn)
            # final fit
            fit_pose(current_frame, last_smpl, frustum, nohands, debug_rn)

            poses_dset[i] = current_frame.smpl.pose.r
            trans_dset[i] = current_frame.smpl.trans.r

            if i == 0:
                betas_dset[:] = current_frame.smpl.betas.r

            last_smpl = current_frame.smpl

    log.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'keypoint_file',
        type=str,
        help="File that contains 2D keypoint detections")
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
        '--model', '-m',
        default='vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        help='Path to SMPL model')
    parser.add_argument(
        '--prior', '-p',
        default='assets/prior_a_pose.pkl',
        help='Path to pose prior')
    parser.add_argument(
        '--resize', '-r', default=0.5, type=float,
        help="Resize factor")
    parser.add_argument(
        '--body_height', '-bh', default=None, type=float,
        help="Height of the subject in meters (optional)")
    parser.add_argument(
        '--nohands', '-nh',
        action='store_true',
        help="Exclude hands from optimization")
    parser.add_argument(
        '--display', '-d',
        action='store_true',
        help="Enable visualization")

    args = parser.parse_args()

    main(args.keypoint_file, args.masks_file, args.camera, args.out, args.model, args.prior, args.resize,
         args.body_height, args.nohands, args.display)
