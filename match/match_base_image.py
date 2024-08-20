#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :match_base_image.py.py
# @Time      :2024/8/20 20:03
import os.path

import cv2
import numpy as np

from entity import Instance
from feature import INSTANCE_INDEX_1, INSTANCE_INDEX_2, INSTANCE_INDEX_3
from feature import INSTANCE_INDEX_4, INSTANCE_INDEX_5, INSTANCE_INDEX_6


def compute_iou_similarity(x: Instance, y: Instance):
    index_list = [
        INSTANCE_INDEX_1,
        INSTANCE_INDEX_2,
        INSTANCE_INDEX_3,
        INSTANCE_INDEX_4,
        INSTANCE_INDEX_5,
        INSTANCE_INDEX_6,
    ]

    score_list = [
        compute_image_iou_similarity(
            x.load_image("contour", idx),
            y.load_image("contour", idx),
        )
        for idx in index_list
    ]
    return sum(score_list) / len(score_list)


def compute_image_iou_similarity(img1, img2):
    img1 = np.array(img1, dtype=np.uint8)
    img2 = np.array(img2, dtype=np.uint8)
    return np.sum(np.logical_and(img1, img2)) / np.sum(np.logical_or(img1, img2))


def compute_hist_similarity(x: Instance, y: Instance):
    index_list = [
        INSTANCE_INDEX_1,
        INSTANCE_INDEX_2,
        INSTANCE_INDEX_3,
        INSTANCE_INDEX_4,
        INSTANCE_INDEX_5,
        INSTANCE_INDEX_6,
    ]

    score_list = [
        compute_image_hist_similarity(
            x.load_image("contour", idx),
            y.load_image("contour", idx),
        )
        for idx in index_list
    ]
    return sum(score_list) / len(score_list)


def compute_image_hist_similarity(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return cv2.compareHist(
        cv2.calcHist([diff], [0], None, [256], [0, 256]),
        cv2.calcHist([img1], [0], None, [256], [0, 256]),
        cv2.HISTCMP_CORREL,
    )


def test_compute_hist_similarity():
    pair_list = [
        ("5492627", "5492658"),
        ("5492568", "5492643"),
        ("5492605", "5492614"),
    ]

    score_list = [
        compute_iou_similarity(
            Instance(os.path.join("../resource/20220820200100", x)),
            Instance(os.path.join("../resource/20220820200100", y)),
        )
        for x, y in pair_list
    ]
    # print(score_list)
    assert score_list[0] > score_list[1] > score_list[2]


if __name__ == '__main__':
    test_compute_hist_similarity()
