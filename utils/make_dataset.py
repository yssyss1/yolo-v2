import sys
sys.path.append("..")

import baker
import json
import numpy as np
import os
from tqdm import tqdm
from utils.data_utils import load_image, parse_annotation
import cv2

labels = ["Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak", "Sail boat", "Swimming person",
          "Flying bird/plane", "Other"]

# def dataset_check(image_dir, xml_dir, labels, name):
#     if not os.path.exists(image_dir):
#         raise FileNotFoundError('{} is not exists'.format(image_dir))
#
#     os.makedirs('./dataset_check', exist_ok=True)
#     print('Start check {} dataset'.format(name))
#
#     instances, _ = parse_annotation(xml_dir, image_dir, labels, name)
#     idx = 0
#     for instance in tqdm(instances, desc='Make npy format {} dataset'.format(name)):
#         idx += 1
#         image_path = instance['filename']
#         image = load_image(image_path)
#
#         for object in instance['object']:
#             cv2.rectangle(image, (object['xmin'], object['ymin']), (object['xmax'], object['ymax']), (0, 255, 0), 2)
#             cv2.putText(image,
#                         object['name'],
#                         (object['xmin'], object['ymin'] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1e-3 * image.shape[0],
#                         (0, 255, 0), 1)
#
#         cv2.imwrite('./dataset_check/{}.jpg'.format(idx), image[..., -1])
#
#     print('End check {} dataset!'.format(name))


def build_npy(image_dir, xml_dir, labels, name):
    if not os.path.exists(image_dir):
        raise FileNotFoundError('{} is not exists'.format(image_dir))

    if not os.path.exists(xml_dir):
        raise FileNotFoundError('{} is not exists'.format(xml_dir))

    print('Start making {} dataset...'.format(name))

    instances, _ = parse_annotation(xml_dir, image_dir, labels, name)
    data_image = []
    data_annotation = []

    for instance in tqdm(instances, desc='Make npy format {} dataset'.format(name)):
        image_path = instance['filename']
        image = load_image(image_path)
        data_image.append(image)
        data_annotation.append(instance)

    np.save('./{}_image.npy'.format(name), data_image)
    np.save('./{}_annotation.npy'.format(name), data_annotation)

    print('End making {} dataset!'.format(name))


@baker.command(
    params={
        "image_dir": "directory which containing image files",
        "xml_dir": "directory which containing xml annotation files",
        "config_path": "configuration file path, json file will be used to parse labels' information",
        "dataset_name": "dataset name",
    }
)
def build_dataset_npy(image_dir, xml_dir, config_path, dataset_name):
    with open(config_path) as config_file:
        config = json.load(config_file)
        build_npy(image_dir, xml_dir, config['labels'], dataset_name)


if __name__ == '__main__':
    baker.run()
