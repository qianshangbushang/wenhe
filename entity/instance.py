import os.path

import cv2


class Instance:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.id = os.path.basename(root_dir)
        self.feature = {}
        return

    def encode(self):
        return self.__dict__

    def format_image_path(self, sub_dir, index, type=""):
        image_name = f"{index}.jpg" if len(type) == 0 else f"{index}_{type}.jpg"
        return f"{self.root_dir}/{sub_dir}/{image_name}"

    def add_feature(self, feature_key, feature_val):
        self.feature[feature_key] = feature_val

    def update_feature(self, f: dict):
        self.feature.update(f)

    def load_image(self, sub_dir, index):
        return cv2.imread(self.format_image_path(sub_dir, index))