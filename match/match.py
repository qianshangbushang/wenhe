#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :match.py.py
# @Time      :2024/8/19 19:31

from argparse import Namespace

from entity import Instance
from .match_base_feature import compute_similarity_base_feature
from .match_base_image import compute_hist_similarity
from .match_base_image import compute_iou_similarity

sim_fn = {
    "feature": compute_similarity_base_feature,
    "iou": compute_iou_similarity,
    "hist": compute_hist_similarity
}


def compute_similarity(x: Instance, y: Instance, args: Namespace = None):
    fn = sim_fn[args.alg]
    return fn(x, y)
