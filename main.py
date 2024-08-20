import json
import os
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import tqdm

from entity import Instance
from feature import AxisFeatureExtractor, AreaFeatureExtractor
from match import compute_similarity


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dir", default="./resource/20220820200100", type=str)
    parser.add_argument("--alg", default="feature", type=str)
    parser.add_argument("--debug", action="store_true", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    result = {}
    for sub_dir in tqdm.tqdm(os.listdir(args.dir)):
        result[sub_dir] = run_one(os.path.join(args.dir, sub_dir), args)
    with open("result.json", "w") as f:
        json.dump(result, f, default=lambda o: o.encode(), indent="    ")
    find_topn_sim(result, 10, args)
    return


def find_topn_sim(features: dict, topn=5, args=None):
    items = list(features.items())
    result = []
    for i, (_, instance_i) in enumerate(items):
        for j, (_, instance_j) in enumerate(items[i + 1:]):
            sim_score = compute_similarity(instance_i, instance_j, args)
            result.append([instance_i, instance_j, sim_score])
    result.sort(key=lambda x: x[2], reverse=True)
    for instance_i, instance_j, score in result[:topn * 2]:
        print(f"item {instance_i.id}, {instance_j.id} match score: {score}")

    for instance1, instance2, _ in result[:topn]:
        id1, id2 = instance1.id, instance2.id
        for idx in range(1, 7):
            img1 = instance1.format_image_path("contour", idx)
            img2 = instance2.format_image_path("contour", idx)

            img1 = cv2.imread(img1)
            img2 = cv2.imread(img2)

            output = cv2.hconcat([img1, img2])

            os.makedirs(f"./output/contrast/{id1}vs{id2}", exist_ok=True)
            output_path = f"./output/contrast/{id1}vs{id2}/{idx}.jpg"
            cv2.imwrite(output_path, output)
    return


def np_int_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_one(root_dir: str, args: Namespace):
    instance = Instance(root_dir)
    extractor = [
        AxisFeatureExtractor(),
        AreaFeatureExtractor(),
    ]
    for e in extractor:
        instance.update_feature(e.extract(instance, args))
    return instance


if __name__ == "__main__":
    main()
