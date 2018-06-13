#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import chumpy as ch

from lib.geometry import visible_boundary_edge_verts
from vendor.smplify.robustifiers import GMOf


def plucker(rays):
    p = rays[:, 0]
    n = rays[:, 1] - rays[:, 0]
    n /= np.linalg.norm(n, axis=1).reshape(-1, 1)
    m = np.cross(p, n, axisa=1, axisb=1)

    return n, m


def distance_function(rays, verts):
    n, m = plucker(rays)
    return ch.cross(verts, n, axisa=1, axisb=1) - m


def unpose_and_select_rays(rays, Vi, smpl, rn_b, rn_m):
    v_ids = visible_boundary_edge_verts(rn_b, rn_m)
    verts = smpl.r[v_ids]

    n, m = plucker(rays)
    dist = np.linalg.norm(np.cross(verts.reshape(-1, 1, 3), n, axisa=2, axisb=1) - m, axis=2)

    ray_matches = np.argmin(dist, axis=0)
    vert_matches = np.argmin(dist, axis=1)

    rays_u_r = np.zeros_like(rays)

    M = Vi[:, :, v_ids]
    T = smpl.v_posevariation[v_ids].r

    tmp0 = M[:, :, ray_matches] * np.hstack((rays[:, 0], np.ones((rays.shape[0], 1)))).T.reshape(1, 4, -1)
    tmp1 = M[:, :, ray_matches] * np.hstack((rays[:, 1], np.ones((rays.shape[0], 1)))).T.reshape(1, 4, -1)

    rays_u_r[:, 0] = np.sum(tmp0, axis=1).T[:, :3] - T[ray_matches]
    rays_u_r[:, 1] = np.sum(tmp1, axis=1).T[:, :3] - T[ray_matches]

    rays_u_v = np.zeros_like(rays[vert_matches])

    tmp0 = M * np.hstack((rays[vert_matches, 0], np.ones((verts.shape[0], 1)))).T.reshape(1, 4, -1)
    tmp1 = M * np.hstack((rays[vert_matches, 1], np.ones((verts.shape[0], 1)))).T.reshape(1, 4, -1)

    rays_u_v[:, 0] = np.sum(tmp0, axis=1).T[:, :3] - T
    rays_u_v[:, 1] = np.sum(tmp1, axis=1).T[:, :3] - T

    valid_rays = dist[np.vstack((ray_matches, range(dist.shape[1]))).tolist()] < 0.12
    valid_verts = dist[np.vstack((range(dist.shape[0]), vert_matches)).tolist()] < 0.03

    ray_matches = ray_matches[valid_rays]

    return np.concatenate((v_ids[ray_matches], v_ids[valid_verts])), \
           np.concatenate((rays_u_r[valid_rays], rays_u_v[valid_verts]))


def rays_from_points(points, camera):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points3d = camera.unproject_points(points)

    c0 = -camera.t.r

    return np.hstack((np.repeat(c0.reshape(1, 1, 3), points3d.shape[0], axis=0), points3d.reshape(-1, 1, 3)))


def rays_from_silh(mask, camera):

    if cv2.__version__[0] == '2':
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    silh = np.zeros_like(mask)

    for con in contours:
        cv2.drawContours(silh, [con], 0, 1, 1)

    points = np.vstack(np.where(silh == 1)[::-1]).astype(np.float32).T
    rays = rays_from_points(points, camera)

    return rays


def ray_objective(f, sigma, base_smpl, camera, vis_rn_b, vis_rn_m):
    base_smpl.pose[:] = f.pose
    camera.t[:] = f.trans

    f.v_ids, f.rays_u = unpose_and_select_rays(f.rays, f.Vi, base_smpl, vis_rn_b, vis_rn_m)
    f.verts = base_smpl.v_shaped_personal[f.v_ids]
    f.dist = distance_function(f.rays_u, f.verts)

    return GMOf(f.dist, sigma)
