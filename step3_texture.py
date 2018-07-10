#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import argparse
import numpy as np
import cPickle as pkl

from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.geometry import VertNormals
from tex.iso import Isomapper, IsoColoredRenderer

from util import im
from util.logger import log
from models.smpl import Smpl


def main(consensus_file, camera_file, video_file, pose_file, masks_file, out, model_file, resolution, num,
         first_frame, last_frame, display):
    # load data
    with open(model_file, 'rb') as fp:
        model_data = pkl.load(fp)

    with open(camera_file, 'rb') as fp:
        camera_data = pkl.load(fp)

    with open(consensus_file, 'rb') as fp:
        consensus_data = pkl.load(fp)

    pose_data = h5py.File(pose_file, 'r')
    poses = pose_data['pose'][first_frame:last_frame]
    trans = pose_data['trans'][first_frame:last_frame]
    masks = h5py.File(masks_file, 'r')['masks'][first_frame:last_frame]
    num_frames = masks.shape[0]
    indices_texture = np.ceil(np.arange(num) * num_frames * 1. / num).astype(np.int)

    vt = np.load('assets/basicModel_vt.npy')
    ft = np.load('assets/basicModel_ft.npy')

    # init
    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = consensus_data['betas']
    base_smpl.v_personal[:] = consensus_data['v_personal']

    bgcolor = np.array([1., 0.2, 1.])
    iso = Isomapper(vt, ft, base_smpl.f, resolution, bgcolor=bgcolor)
    iso_vis = IsoColoredRenderer(vt, ft, base_smpl.f, resolution)
    camera = ProjectPoints(t=camera_data['camera_t'], rt=camera_data['camera_rt'], c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)
    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}
    rn_vis = ColoredRenderer(f=base_smpl.f, frustum=frustum, camera=camera, num_channels=1)

    cap = cv2.VideoCapture(video_file)
    for _ in range(first_frame):
        cap.grab()

    # get part-textures
    i = first_frame

    tex_agg = np.zeros((resolution, resolution, 25, 3))
    tex_agg[:] = np.nan
    normal_agg = np.ones((resolution, resolution, 25)) * 0.2

    vn = VertNormals(f=base_smpl.f, v=base_smpl)
    static_indices = np.indices((resolution, resolution))

    while cap.isOpened() and i < indices_texture[-1]:
        if i in indices_texture:
            log.info('Getting part texture from frame {}...'.format(i))
            _, frame = cap.read()

            mask = np.array(masks[i], dtype=np.uint8)
            pose_i = np.array(poses[i], dtype=np.float32)
            trans_i = np.array(trans[i], dtype=np.float32)

            base_smpl.pose[:] = pose_i
            base_smpl.trans[:] = trans_i

            # which faces have been seen and are projected into the silhouette?
            visibility = rn_vis.visibility_image.ravel()
            visible = np.nonzero(visibility != 4294967295)[0]

            proj = camera.r
            in_viewport = np.logical_and(
                np.logical_and(np.round(camera.r[:, 0]) >= 0, np.round(camera.r[:, 0]) < frustum['width']),
                np.logical_and(np.round(camera.r[:, 1]) >= 0, np.round(camera.r[:, 1]) < frustum['height']),
            )
            in_mask = np.zeros(camera.shape[0], dtype=np.bool)
            idx = np.round(proj[in_viewport][:, [1, 0]].T).astype(np.int).tolist()
            in_mask[in_viewport] = mask[idx]

            faces_in_mask = np.where(np.min(in_mask[base_smpl.f], axis=1))[0]
            visible_faces = np.intersect1d(faces_in_mask, visibility[visible])

            # get the current unwrap
            part_tex = iso.render(frame / 255., camera, visible_faces)

            # angle under which the texels have been seen
            points = np.hstack((proj, np.ones((proj.shape[0], 1))))
            points3d = camera.unproject_points(points)
            points3d /= np.linalg.norm(points3d, axis=1).reshape(-1, 1)
            alpha = np.sum(points3d * -vn.r, axis=1).reshape(-1, 1)
            alpha[alpha < 0] = 0
            iso_normals = iso_vis.render(alpha)[:, :, 0]
            iso_normals[np.all(part_tex == bgcolor, axis=2)] = 0

            # texels to consider
            part_mask = np.zeros((resolution, resolution))
            min_normal = np.min(normal_agg, axis=2)
            part_mask[iso_normals > min_normal] = 1.

            # update best seen texels
            where = np.argmax(np.atleast_3d(iso_normals) - normal_agg, axis=2)

            idx = np.dstack((static_indices[0], static_indices[1], where))[part_mask == 1]
            tex_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = part_tex[part_mask == 1]
            normal_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = iso_normals[part_mask == 1]

            if display:
                im.show(part_tex, id='part_tex', waittime=1)

        else:
            cap.grab()

        i += 1

    # merge textures
    log.info('Computing median texture...')
    tex_median = np.nanmedian(tex_agg, axis=2)

    log.info('Inpainting unseen areas...')
    where = np.max(normal_agg, axis=2) > 0.2

    tex_mask = iso.iso_mask
    mask_final = np.float32(where)

    kernel_size = np.int(resolution * 0.02)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    inpaint_area = cv2.dilate(tex_mask, kernel) - mask_final

    tex_final = cv2.inpaint(np.uint8(tex_median * 255), np.uint8(inpaint_area * 255), 3, cv2.INPAINT_TELEA)

    cv2.imwrite(out, tex_final)
    log.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'consensus',
        type=str,
        help="pkl file that contains consensus")
    parser.add_argument(
        'camera',
        type=str,
        help="pkl file that contains camera settings")
    parser.add_argument(
        'video',
        type=str,
        help="Input video")
    parser.add_argument(
        'pose_file',
        type=str,
        help="File that contains poses")
    parser.add_argument(
        'masks_file',
        type=str,
        help="File that contains segmentations")
    parser.add_argument(
        'out',
        type=str,
        help="Out file path")
    parser.add_argument(
        '--model', '-m',
        default='vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        help='Path to SMPL model')
    parser.add_argument(
        '--resolution', '-r', default=1000, type=int,
        help="Output resolution")
    parser.add_argument(
        '--num', '-n', default=120, type=int,
        help="Number of used frames")
    parser.add_argument(
        '--first_frame', '-f', default=0, type=int,
        help="First frame to use")
    parser.add_argument(
        '--last_frame', '-l', default=2000, type=int,
        help="Last frame to use")
    parser.add_argument(
        '--display', '-d',
        action='store_true',
        help="Enable visualization")

    args = parser.parse_args()

    main(args.consensus, args.camera, args.video, args.pose_file, args.masks_file, args.out, args.model,
         args.resolution, args.num, args.first_frame, args.last_frame, args.display)
