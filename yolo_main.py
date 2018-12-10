# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import baker
import json
from model.yolo import YOLO


@baker.command(
    params={
        "config_path": "configuration file path",
    }
)
def train(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.train()


@baker.command(
    params={
        "weight_path": "trained weight path - weight path which will be used for prediction",
        "image_path": "image path",
        "config_path": "configuration file path",
        "obj_threshold": "obj threshold (confidence * probability) - default: 0.3",
        "nms_threshold": "nms threshold (threshold for non maximum suppression)- default: 0.3"
    }
)
def inference(weight_path, image_path, config_path, obj_threshold=0.3, nms_threshold=0.3):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.inference(weight_path, image_path, float(obj_threshold), float(nms_threshold))


@baker.command(
    params={
        "weight_path": "trained weight path - weight path which will be used for mAP evaluation",
        "config_path": "configuration file path",
        "iou_threshold": "mAP evalution threshold - default: 0.5",
        "obj_threshold": "obj threshold (confidence * probability) - default: 0.3",
        "nms_threshold": "nms threshold (threshold for non maximum suppression)- default: 0.3"
    }
)
def evaluate(weight_path, config_path, iou_threshold=0.5, obj_threshold=0.3, nms_threshold=0.3):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.mAP_evalutation(weight_path, float(iou_threshold), float(obj_threshold), float(nms_threshold))


@baker.command(
    params={
        "weight_path": "trained weight path - weight path which will be used for prediction",
        "config_path": "configuration file path",
        "obj_threshold": "obj threshold (confidence * probability) - default: 0.3",
        "nms_threshold": "nms threshold (threshold for non maximum suppression)- default: 0.3"
    }
)
def show_me_camera(weight_path, config_path, obj_threshold=0.3, nms_threshold=0.3):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.show_me_camera(weight_path, float(obj_threshold), float(nms_threshold))


@baker.command(
    params={
        "weight_path": "trained weight path - weight path which will be used for prediction",
        "config_path": "configuration file path",
        "save_path": "directory in which result images are saved",
        "obj_threshold": "obj threshold (confidence * probability) - default: 0.3",
        "nms_threshold": "nms threshold (threshold for non maximum suppression)- default: 0.3"
    }
)
def inference_valid(weight_path, config_path,  save_path='./results/valid', obj_threshold=0.3, nms_threshold=0.3):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.inference_valid(weight_path, save_path, float(obj_threshold), float(nms_threshold))


if __name__ == '__main__':
    baker.run()
