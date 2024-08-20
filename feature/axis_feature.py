#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :axis_feature.py
# @Time      :2024/8/19 19:24

from .base import *


class AxisFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.feature_key = "f_axis"
        return

    def extract(self, instance: Instance, args: Namespace):
        return {
            self.feature_key:
                self.find_limit_point(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_1), args) +
                self.find_limit_point(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_2), args) +
                self.find_limit_point(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_3), args) +
                self.find_limit_point(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_4), args) +
                self.find_limit_point(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_5), args) +
                self.find_limit_point(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_6), args),
        }

    def find_limit_point(self, image_path, args: Namespace):
        img = self.load_image(image_path, args)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        b, c, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return [
            self.find_point(c[0], POINT_LEFT),
            self.find_point(c[0], POINT_RIGHT),
            self.find_point(c[0], POINT_TOP),
            self.find_point(c[0], POINT_BOTTOM)
        ]

    def find_point(self, contour, point_type):
        if point_type == POINT_LEFT:
            return min([int(p[0][0]) for p in contour])
        if point_type == POINT_RIGHT:
            return max([int(p[0][0]) for p in contour])
        if point_type == POINT_TOP:
            return min([int(p[0][1]) for p in contour])
        if point_type == POINT_BOTTOM:
            return max([int(p[0][1]) for p in contour])
