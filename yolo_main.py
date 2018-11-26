import baker
import json
from model.yolo import YOLO


@baker.command(
    params={
        "config_path": "configuration file path - default: ./config/yolo.json",
    }
)
def train(config_path='./config/yolo.json'):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.train()


@baker.command(
    params={
        "weight_path": "trained weight path - inference에 사용할 weight의 경로를 명시해줘야함",
        "image_path": "image path",
        "config_path": "configuration file path - default: ./config/yolo.json",
        "obj_threshold": "obj threshold (confidence * probability) - default: 0.3",
        "nms_threshold": "nms threshold (겹친 이미지를 제거하기 위한 임계치)- default: 0.3"
    }
)
def inference(weight_path, image_path, config_path='./config/yolo.json', obj_threshold=0.3, nms_threshold=0.3):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.inference(weight_path, image_path, float(obj_threshold), float(nms_threshold))


@baker.command(
    params={
        "weight_path": "trained weight path - mAP evalution에 사용할 weight의 경로를 명시해줘야함",
        "iou_threshold": "mAP evalution의 threshold - default: 0.5",
        "config_path": "configuration file path - default: ./config/yolo.json",
    }
)
def evaluate(weight_path, iou_threshold=0.5, config_path='./config/yolo.json'):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.mAP_evalutation(float(iou_threshold), weight_path)


@baker.command(
    params={
        "weight_path": "trained weight path - inference에 사용할 weight의 경로를 명시해줘야함",
        "config_path": "configuration file path - default: ./config/yolo.json",
        "obj_threshold": "obj threshold (confidence * probability) - default: 0.3",
        "nms_threshold": "nms threshold (겹친 이미지를 제거하기 위한 임계치)- default: 0.3"
    }
)
def show_me_camera(weight_path, config_path='./config/yolo.json', obj_threshold=0.3, nms_threshold=0.3):
    with open(config_path) as config_file:
        config = json.load(config_file)
        yolo = YOLO(config)
        yolo.show_me_camera(weight_path, float(obj_threshold), float(nms_threshold))


if __name__ == '__main__':
    baker.run()