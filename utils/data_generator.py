from imgaug import augmenters as iaa
from keras.utils import Sequence
import copy
from utils.data_utils import BoundBox, bbox_iou, normalize
import cv2
import numpy as np


class BatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True, augmentation=True, norm=True):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config["ANCHORS"][2 * i], config["ANCHORS"][2 * i + 1]) for i in
                        range(int(len(config["ANCHORS"]) // 2))]

        # https://github.com/aleju/imgaug에서 Data Augmentation Function들 참고할 것.
        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0, 5),
                           [
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           ],
                           random_order=True
                           )
            ]
        )

        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config["BATCH_SIZE"]))

    def num_classes(self):
        return len(self.config["LABELS"])

    def size(self):
        return len(self.images)

    def __getitem__(self, idx):
        l_bound = idx * self.config["BATCH_SIZE"]
        r_bound = (idx + 1) * self.config["BATCH_SIZE"]

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config["BATCH_SIZE"]

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config["IMAGE_H"], self.config["IMAGE_W"], 3))
        b_batch = np.zeros((r_bound - l_bound, self.config["GRID_H"], self.config["GRID_W"],
                            self.config["TRUE_BOX_BUFFER"], 4 + 1 + len(self.config["LABELS"])))
        y_batch = np.zeros((r_bound - l_bound, self.config["GRID_H"], self.config["GRID_W"], self.config["BOX"],
                            4 + 1 + len(self.config["LABELS"])))

        for train_instance in self.images[l_bound:r_bound]:
            img, all_objs = self.get_image_with_box(train_instance, augmentation=self.augmentation)

            true_box_index = 0

            for obj in all_objs:
                if obj["xmax"] > obj["xmin"] and obj["ymax"] > obj["ymin"] and obj["name"] in self.config["LABELS"]:
                    center_x = .5 * (obj["xmin"] + obj["xmax"])
                    center_x = center_x / (float(self.config["IMAGE_W"]) / self.config["GRID_W"])
                    center_y = .5 * (obj["ymin"] + obj["ymax"])
                    center_y = center_y / (float(self.config["IMAGE_H"]) / self.config["GRID_H"])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config["GRID_W"] and grid_y < self.config["GRID_H"]:
                        obj_indx = self.config["LABELS"].index(obj["name"])

                        center_w = (obj["xmax"] - obj["xmin"]) / (
                                float(self.config["IMAGE_W"]) / self.config["GRID_W"])
                        center_h = (obj["ymax"] - obj["ymin"]) / (
                                float(self.config["IMAGE_H"]) / self.config["GRID_H"])

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
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        b_batch[instance_count, 0, 0, true_box_index, :4] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config["TRUE_BOX_BUFFER"]

            x_batch[instance_count] = normalize(img) if self.norm else img

            instance_count += 1

        return x_batch, np.concatenate([b_batch, y_batch], axis=-2)

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def get_image_with_box(self, train_instances, augmentation):
        image_name = train_instances["filename"]
        image = cv2.imread(image_name)

        if image is None:
            raise FileNotFoundError("Cannot find image {}".format(image_name))

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instances["object"])

        if augmentation:

            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, None, fx=scale, fy=scale)

            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        image = cv2.resize(image, (self.config["IMAGE_H"], self.config["IMAGE_W"]))
        image = image[:, :, ::-1] # opencv는 채널이 BGR이므로 RBG로 채널 변환

        for obj in all_objs:
            for attr in ["xmin", "xmax"]:
                if augmentation:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config["IMAGE_W"]) / w)
                obj[attr] = max(min(obj[attr], self.config["IMAGE_W"]), 0)

            for attr in ["ymin", "ymax"]:
                if augmentation:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config["IMAGE_H"]) / h)
                obj[attr] = max(min(obj[attr], self.config["IMAGE_H"]), 0)

            if augmentation and flip > 0.5:
                xmin = obj["xmin"]
                obj["xmin"] = self.config["IMAGE_W"] - obj["xmax"]
                obj["xmax"] = self.config["IMAGE_W"] - xmin

        return image, all_objs
