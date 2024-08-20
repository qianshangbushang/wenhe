#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :match_base_feature.py.py
# @Time      :2024/8/20 20:01


import math
from functools import reduce

from entity import Instance


def compute_area_similarity(x, y):
    diff_rev = [1 / math.fabs(i - j + 1e-5) for i, j in zip(x, y)]
    return reduce(lambda t, s: t * s, diff_rev, 1)


def compute_axis_similarity(x, y):
    diff_rev = [1 / math.fabs(i - j + 1e-5) for i, j in zip(x, y)]
    return reduce(lambda t, s: t * s, diff_rev, 1)


def compute_similarity_base_feature(x: Instance, y: Instance):
    res = []
    for key in x.feature.keys() & y.feature.keys():
        if key == "f_area":
            res.append(compute_area_similarity(x.feature[key], y.feature[key]))
        if key == "f_axis":
            res.append(compute_axis_similarity(x.feature[key], y.feature[key]))
    return reduce(lambda t, s: t * s, res, 1.0)


def test_compute_are_similarity():
    x = [2, 2, 4, 4, 2]
    y = [4, 4, 2, 2, 4]

    assert compute_area_similarity(x, y) - 1 / 32 <= 1e-5
    return
