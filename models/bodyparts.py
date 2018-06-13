#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cPickle as pkl
import numpy as np

_cache = None


def get_bodypart_vertex_ids():
    global _cache

    if _cache is None:
        with open('assets/bodyparts.pkl', 'rb') as fp:
            _cache = pkl.load(fp)

    return _cache


def faces_no_hands(f):
    v_ids = get_bodypart_vertex_ids()
    hands = np.concatenate((v_ids['hand_r'], v_ids['hand_l']))
    return np.array(filter(lambda face: np.intersect1d(face, hands).size == 0, f))


def regularize_laplace():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 12.
    reg[v_ids['hand_l']] = 5.
    reg[v_ids['hand_r']] = 5.
    reg[v_ids['fingers_l']] = 8.
    reg[v_ids['fingers_r']] = 8.
    reg[v_ids['foot_l']] = 5.
    reg[v_ids['foot_r']] = 5.
    reg[v_ids['toes_l']] = 8.
    reg[v_ids['toes_r']] = 8.
    reg[v_ids['ear_l']] = 10.
    reg[v_ids['ear_r']] = 10.

    return reg


def regularize_model():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 7.
    reg[v_ids['hand_l']] = 12.
    reg[v_ids['hand_r']] = 12.
    reg[v_ids['fingers_l']] = 15.
    reg[v_ids['fingers_r']] = 15.
    reg[v_ids['foot_l']] = 12.
    reg[v_ids['foot_r']] = 12.
    reg[v_ids['toes_l']] = 15.
    reg[v_ids['toes_r']] = 15.
    reg[v_ids['ear_l']] = 10.
    reg[v_ids['ear_r']] = 10.

    return reg


def regularize_symmetry():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 10.
    reg[v_ids['hand_l']] = 10.
    reg[v_ids['hand_r']] = 10.
    reg[v_ids['foot_l']] = 10.
    reg[v_ids['foot_r']] = 10.
    reg[v_ids['ear_l']] = 5.
    reg[v_ids['ear_r']] = 5.

    return reg
