from imgaug import augmenters as iaa
from keras.utils import Sequence
import copy
from utils.data_utils import BoundBox, bbox_iou, normalize
import cv2
import numpy as np


class BatchGenerator(Sequence):
    def __init__(self, images, annotations, config, shuffle=True, augmentation=True, norm=True):
        super(BatchGenerator, self).__init__()
        self.generator = None

        self.images = images
        self.annotations = annotations
        self.config = config

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.norm = norm
        self.scale_factor = 0
        self.batch_idx = np.arange(len(self.annotations))

        self.anchors = [BoundBox(0, 0, config["ANCHORS"][2 * i], config["ANCHORS"][2 * i + 1]) for i in
                        range(int(len(config["ANCHORS"]) // 2))]

        # https://github.com/aleju/imgaug
        self.aug_pipe = iaa.Sequential(
            [
                iaa.Sometimes(0.2, [iaa.GaussianBlur((0, 2.0))]),
                iaa.Sometimes(0.5, [iaa.Add((-25, 25))]),
            ]

        )

        if shuffle:
            np.random.shuffle(self.batch_idx)

    def __len__(self):
        return int(np.ceil(float(len(self.annotations)) / self.config["BATCH_SIZE"]))

    def num_classes(self):
        return len(self.config["LABELS"])

    def size(self):
        return len(self.annotations)

    def load_image(self, i):
        return self.images[i]

    def load_annotation(self, i):
        annots = []

        for obj in self.annotations[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def __getitem__(self, idx):
        l_bound = idx * self.config["BATCH_SIZE"]
        r_bound = (idx + 1) * self.config["BATCH_SIZE"]

        if idx % 10 == 0:
            self.scale_factor = np.random.randint(-3, 7, 1)[0] if self.config["MULTI_SCALE_TRAINING"] else 0

        image_width = self.config["IMAGE_W"] + (self.config["IMAGE_W"] // self.config["GRID_W"] * self.scale_factor)
        image_height = self.config["IMAGE_H"] + (self.config["IMAGE_H"] // self.config["GRID_H"] * self.scale_factor)

        grid_w = self.config["GRID_W"] + self.scale_factor
        grid_h = self.config["GRID_H"] + self.scale_factor

        grid_dims = np.tile(np.array([grid_h, grid_w]), (self.config["BATCH_SIZE"], 1))

        if r_bound > len(self.annotations):
            r_bound = len(self.annotations)
            l_bound = r_bound - self.config["BATCH_SIZE"]

        instance_count = 0

        x_batch = np.zeros((self.config["BATCH_SIZE"], image_height, image_width, 3))
        y_batch = np.zeros((self.config["BATCH_SIZE"], grid_h, grid_w, self.config["BOX"],
                            4 + 1 + self.config["CLASS"]))

        for idx in self.batch_idx[l_bound:r_bound]:
            batch_image = self.images[idx]
            batch_annotations = self.annotations[idx]

            img, all_objs = self.get_image_with_box(batch_image, batch_annotations, image_height, image_width, augmentation=self.augmentation)

            for obj in all_objs:
                if obj["xmax"] > obj["xmin"] and obj["ymax"] > obj["ymin"] and obj["name"] in self.config["LABELS"]:
                    center_x = .5 * (obj["xmin"] + obj["xmax"])
                    center_x = center_x / (float(image_width) / grid_w)
                    center_y = .5 * (obj["ymin"] + obj["ymax"])
                    center_y = center_y / (float(image_height) / grid_h)

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < grid_w and grid_y < grid_h:
                        class_idx = self.config["LABELS"].index(obj["name"])

                        center_w = (obj["xmax"] - obj["xmin"]) / (
                                float(image_width) / grid_w)
                        center_h = (obj["ymax"] - obj["ymin"]) / (
                                float(image_height) / grid_h)

                        box = [center_x, center_y, center_w, center_h]

                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + class_idx] = 1

            x_batch[instance_count] = normalize(img) if self.norm else img
            instance_count += 1

        y_batch = np.reshape(y_batch, (self.config["BATCH_SIZE"], grid_h, grid_w, self.config["BOX"]*(4 + 1 + self.config["CLASS"])))
        return [x_batch, grid_dims], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_idx)

    def get_image_with_box(self, image, annotations, image_height, image_width, augmentation):
        h, w, c = image.shape
        all_objs = copy.deepcopy(annotations["object"])

        if augmentation:

            scale = np.random.uniform() / 5. + 1.
            image = cv2.resize(image, None, fx=scale, fy=scale)

            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            # TODO Add Data augmentation - Rotation
            image = self.aug_pipe.augment_image(image)

        image = cv2.resize(image, (image_height, image_width))

        for obj in all_objs:
            for attr in ["xmin", "xmax"]:
                if augmentation:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(image_width) / w)
                obj[attr] = max(min(obj[attr], image_width), 0)

            for attr in ["ymin", "ymax"]:
                if augmentation:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(image_height) / h)
                obj[attr] = max(min(obj[attr], image_height), 0)

            if augmentation and flip > 0.5:
                xmin = obj["xmin"]
                obj["xmin"] = image_width - obj["xmax"]
                obj["xmax"] = image_width - xmin

        return image, all_objs
