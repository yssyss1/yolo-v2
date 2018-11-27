import sys
sys.path.append("..")

import numpy as np
import os
from tqdm import tqdm
from utils.data_utils import load_image, parse_annotation, draw_boxes, BoundBox
import cv2


def dataset_check(image_dir, xml_dir, labels, name):
    if not os.path.exists(image_dir):
        raise FileNotFoundError('{} is not exists'.format(image_dir))

    os.makedirs('./dataset_check', exist_ok=True)
    print('Start check {} dataset'.format(name))

    instances, _ = parse_annotation(xml_dir, image_dir, labels, name)
    idx = 0
    for instance in tqdm(instances, desc='Make npy format {} dataset'.format(name)):
        idx += 1
        image_path = instance['filename']
        image = load_image(image_path)

        for object in instance['object']:
            cv2.rectangle(image, (object['xmin'], object['ymin']), (object['xmax'], object['ymax']), (0, 255, 0), 2)
            cv2.putText(image,
                        object['name'],
                        (object['xmin'], object['ymin'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0, 255, 0), 1)

        cv2.imwrite('./dataset_check/{}.jpg'.format(idx), image[..., -1])

    print('End check {} dataset!'.format(name))


def build_npy(image_dir, xml_dir, labels, name):
    if not os.path.exists(image_dir):
        raise FileNotFoundError('{} is not exists'.format(image_dir))

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


if __name__ == '__main__':
    labels = ["Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak", "Sail boat", "Swimming person",
              "Flying bird/plane", "Other"]
    dataset_check('./singapore_maritime/train/image', './singapore_maritime/train/annotation', labels, 'valid')
