import os
import cv2
import time
import numpy as np
from keras.models import load_model
import tensorflow as tf

label = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"]


def show_me_camera(model_path, obj_threshold=0.3, nms_threshold=0.3):
    if not os.path.exists(model_path):
        raise FileNotFoundError('{} is not exists'.format(model_path))

    print('Load weight from {}'.format(model_path))
    model = load_model(model_path, custom_objects={'tf': tf})

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise NotImplemented('Unable to read camera feed')

    print('Video capture start!')

    while True:
        start = time.time()
        ret, frame = cap.read()

        if ret:
            input_image = cv2.resize(frame, (416, 416))
            input_image = input_image / 255.
            input_image = np.expand_dims(input_image, 0)

            netout = model.predict(input_image)

            boxes = decode_netout(netout[0],
                                  shape_dims=(13, 13, 5, 4 + 1 + 80),
                                  obj_threshold=obj_threshold,
                                  nms_threshold=nms_threshold,
                                  anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                                  nb_class=80)

            frame = draw_boxes(frame, boxes, 13, 13, labels=label)
            end = time.time()
            seconds = end - start
            print(seconds)
            fps = 1. / seconds

            cv2.putText(frame,
                        "fps : {}".format(int(fps)),
                        (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0), 1)

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Video capture end!')


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


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def bbox_iou(box1, box2):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


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


if __name__ == '__main__':
    show_me_camera('/home/seok/yolo_model.h5')