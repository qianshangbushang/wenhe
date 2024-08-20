#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :area_feature.py.py
# @Time      :2024/8/19 19:22

from .base import *


class AreaFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.feature_key = "f_area"
        return

    def extract(self, instance: Instance, args: Namespace):
        return {
            self.feature_key: [
                self.compute_area(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_1), args),
                self.compute_area(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_2), args),
                self.compute_area(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_3), args),
                self.compute_area(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_4), args),
                self.compute_area(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_5), args),
                self.compute_area(instance.format_image_path(INSTANCE_SUBDIR_CONTOUR, INSTANCE_INDEX_6), args),
            ]
        }

    def compute_area(self, image_path, args: Namespace):
        img = self.load_image(image_path, args)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        b, c, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if args.debug:
            draw_img = img.copy()
            draw_img = cv2.drawContours(draw_img, c[0:1], -1, (0, 0, 255), 2)
            cv2.imshow("contour", draw_img)
            cv2.waitKey(5000)
        return cv2.contourArea(c[0])
