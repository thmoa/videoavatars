#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import logging

__all__ = ['log']


FORMAT = "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=FORMAT)
