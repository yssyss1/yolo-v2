import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.confidence = confidence
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def normalize(image):
    return image / 255.


def bbox_iou(box1: BoundBox, box2: BoundBox):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def parse_annotation(ann_dir, img_dir, labels, data_name):
    if len(labels) == 0:
        raise ValueError("given label is not valid")

    print("Start Parsing {} data annotions...".format(data_name))

    all_imgs = []
    seen_labels = {}

    for ann in tqdm(sorted(os.listdir(ann_dir)), desc="Parse {} annotations".format(data_name)):
        img = {"object": []}

        tree = ET.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if "filename" in elem.tag:
                img["filename"] = os.path.join(img_dir, elem.text)
            if "width" in elem.tag:
                img["width"] = int(elem.text)
            if "height" in elem.tag:
                img["height"] = int(elem.text)
            if "object" in elem.tag or "part" in elem.tag:
                obj = {}

                for attr in list(elem):
                    if "name" in attr.tag:
                        obj["name"] = attr.text

                        if obj["name"] in seen_labels:
                            seen_labels[obj["name"]] += 1
                        else:
                            seen_labels[obj["name"]] = 1

                        if len(labels) > 0 and obj["name"] not in labels:
                            break
                        else:
                            img["object"] += [obj]

                    if "bndbox" in attr.tag:
                        for dim in list(attr):
                            if "xmin" in dim.tag:
                                obj["xmin"] = int(round(float(dim.text)))
                            if "ymin" in dim.tag:
                                obj["ymin"] = int(round(float(dim.text)))
                            if "xmax" in dim.tag:
                                obj["xmax"] = int(round(float(dim.text)))
                            if "ymax" in dim.tag:
                                obj["ymax"] = int(round(float(dim.text)))

        if len(img["object"]) > 0:
            all_imgs += [img]

    print("End Parsing Annotations!")

    return all_imgs, seen_labels


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def decode_netout(netout, shape_dims, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    netout = np.reshape(netout, shape_dims)
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):

                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    x, y, w, h, confidence = netout[row, col, b, :5]

                    x = (col + sigmoid(x))
                    y = (row + sigmoid(y))
                    w = anchors[2 * b + 0] * np.exp(w)
                    h = anchors[2 * b + 1] * np.exp(h)

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)

                    boxes.append(box)

    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    boxes = [box for box in boxes if box.get_score() > 0]

    return boxes


def draw_boxes(image, boxes, grid_h, grid_w, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = max(int(box.xmin * image_w / grid_w), 0)
        ymin = max(int(box.ymin * image_h / grid_h), 0)
        xmax = min(int(box.xmax * image_w / grid_w), image_w)
        ymax = min(int(box.ymax * image_h / grid_h), image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image,
                    labels[box.get_label()] + " " + str(box.get_score()),
                    (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (0, 255, 0), 1)
    return image


def load_image(image_path):
    image = cv2.imread(image_path)
    image = np.array(image[:, :, ::-1])

    return image


def compute_overlap(a, b):
    bounding_box_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    annotation_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    intersect_width = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    intersect_height = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])

    intersect_width = np.maximum(intersect_width, 0)
    intersect_height = np.maximum(intersect_height, 0)

    intersect_area = intersect_width * intersect_height

    union_area = bounding_box_area + annotation_area - intersect_area

    return intersect_area / union_area


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    average_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return average_precision


def load_npy(file_path):
    return np.load(file_path)
