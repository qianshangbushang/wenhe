from abc import ABC
from argparse import Namespace

import cv2

from entity import Instance

INSTANCE_SUBDIR_CONTOUR = "contour"
INSTANCE_SUBDIR_CUT = "cut"
INSTANCE_SUBDIR_DETAIL = "detail"
INSTANCE_SUBDIR_THRESH = "thresh"

INSTANCE_INDEX_1 = "1"
INSTANCE_INDEX_2 = "2"
INSTANCE_INDEX_3 = "3"
INSTANCE_INDEX_4 = "4"
INSTANCE_INDEX_5 = "5"
INSTANCE_INDEX_6 = "6"

POINT_LEFT = "left"
POINT_RIGHT = "right"
POINT_TOP = "top"
POINT_BOTTOM = "bottom"


class FeatureExtractor(ABC):
    def extract(self, instance: Instance, args: Namespace):
        pass

    def load_image(self, path: str, args: Namespace):
        return cv2.imread(path, )
