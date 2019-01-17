from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import numpy as np
from utils.data_generator import BatchGenerator
import os
import matplotlib.pyplot as plt
import cv2
from utils.data_utils import decode_netout, draw_boxes, load_image, compute_overlap, compute_ap, load_npy
from tqdm import tqdm
import time


class YOLO:

    def __init__(self, config):

        self.config = config
        self.train_image_path = self.__set_variable('train_image_path', '', config)
        self.train_annotation_path = self.__set_variable('train_annotation_path', '', config)
        self.valid_image_path = self.__set_variable('valid_image_path', '', config)
        self.valid_annotation_path = self.__set_variable('valid_annotation_path', '', config)
        self.image_height = self.__set_variable('image_height', 416, config)
        self.image_width = self.__set_variable('image_width', 416, config)
        self.box_num = self.__set_variable('box_num', 5, config)
        self.grid_h = self.__set_variable('grid_h', 13, config)
        self.grid_w = self.__set_variable('grid_w', 13, config)
        self.coord_scale = self.__set_variable('coord_scale', 5.0, config)
        self.object_scale = self.__set_variable('object_scale', 5.0, config)
        self.no_object_scale = self.__set_variable('no_object_scale', 0.5, config)
        self.batch_size = self.__set_variable('batch_size', 32, config)
        self.pretrained_weight = self.__set_variable('pretrained_weight', None, config)
        self.labels = self.__set_variable('labels', None, config)
        self.anchors = self.__set_variable('anchors', None, config)
        self.min_lr = self.__set_variable('min_lr', 1E-7, config)
        self.lr = self.__set_variable('lr', 0.5E-4, config)
        self.lr_decay_rate = self.__set_variable('lr_decay_rate', 0.5, config)
        self.result_path = self.__set_variable('save_path', './results', config)
        self.epoch = self.__set_variable('epoch', 100, config)
        self.multi_scale_training = self.__set_variable('multi_scale_training', False, config)

        if self.labels is None or self.anchors is None:
            raise ValueError('Label and Anchor must be specified in yolo.json file')

        self.class_num = len(self.labels)

        self.model = self.build_model()

    def build_model(self, model_compile=True):
        print("Start Building Model...")

        def conv_block(filters, kernel_size, strides, idx, padding="same", use_bias=False, use_batchnorm=True,
                       use_maxpool=True, pool_size=(2, 2), alpha=0.1):
            def _conv_block(x):
                x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                           name="conv_{}".format(idx), use_bias=use_bias)(x)
                x = BatchNormalization(name="norm_{}".format(idx))(x) if use_batchnorm else x
                x = LeakyReLU(alpha=alpha)(x)
                x = MaxPooling2D(pool_size=pool_size)(x) if use_maxpool else x

                return x

            return _conv_block

        input_image = Input(shape=(None, None, 3))
        grid_dims = Input(shape=(2, ), dtype=tf.int32)

        x = conv_block(filters=32, kernel_size=(3, 3), strides=(1, 1), idx=1)(input_image)
        x = conv_block(filters=64, kernel_size=(3, 3), strides=(1, 1), idx=2)(x)
        x = conv_block(filters=128, kernel_size=(3, 3), strides=(1, 1), idx=3, use_maxpool=False)(x)
        x = conv_block(filters=64, kernel_size=(1, 1), strides=(1, 1), idx=4, use_maxpool=False)(x)
        x = conv_block(filters=128, kernel_size=(3, 3), strides=(1, 1), idx=5)(x)
        x = conv_block(filters=256, kernel_size=(3, 3), strides=(1, 1), idx=6, use_maxpool=False)(x)
        x = conv_block(filters=128, kernel_size=(1, 1), strides=(1, 1), idx=7, use_maxpool=False)(x)
        x = conv_block(filters=256, kernel_size=(3, 3), strides=(1, 1), idx=8)(x)
        x = conv_block(filters=512, kernel_size=(3, 3), strides=(1, 1), idx=9, use_maxpool=False)(x)
        x = conv_block(filters=256, kernel_size=(1, 1), strides=(1, 1), idx=10, use_maxpool=False)(x)
        x = conv_block(filters=512, kernel_size=(3, 3), strides=(1, 1), idx=11, use_maxpool=False)(x)
        x = conv_block(filters=256, kernel_size=(1, 1), strides=(1, 1), idx=12, use_maxpool=False)(x)
        x = conv_block(filters=512, kernel_size=(3, 3), strides=(1, 1), idx=13, use_maxpool=False)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=14, use_maxpool=False)(x)
        x = conv_block(filters=512, kernel_size=(1, 1), strides=(1, 1), idx=15, use_maxpool=False)(x)
        x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=16, use_maxpool=False)(x)
        x = conv_block(filters=512, kernel_size=(1, 1), strides=(1, 1), idx=17, use_maxpool=False)(x)
        x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=18, use_maxpool=False)(x)

        x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=19, use_maxpool=False)(x)
        x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=20, use_maxpool=False)(x)

        skip_connection = conv_block(filters=64, kernel_size=(1, 1), strides=(1, 1), idx=21, use_maxpool=False)(
            skip_connection)
        skip_connection = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(skip_connection)

        x = concatenate([skip_connection, x])

        x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=22, use_maxpool=False)(x)
        x = Conv2D(self.box_num * (4 + 1 + self.class_num), (1, 1), strides=(1, 1), padding="same", name="conv_23", kernel_initializer='glorot_normal')(x)

        output = x

        # DEPRECATED
        # output = Reshape((self.grid_h, self.grid_w, self.box_num, 4 + 1 + self.class_num))(x)

        model = Model([input_image, grid_dims], output)
        model.summary()

        # TODO - Only for training
        if model_compile:
            model.compile(loss=self._compile_loss(grid_dims), optimizer=RMSprop(lr=self.lr))

        print("End Building Model...")
        return model

    def _compile_loss(self, grid_dims):

        def yolo_loss(y_true, y_pred):

            grid_h = grid_dims[:, 0][0]
            grid_w = grid_dims[:, 1][0]

            y_pred = tf.reshape(y_pred, (self.batch_size, grid_h, grid_w, self.box_num, 4 + 1 + self.class_num))
            y_true = tf.reshape(y_true, (self.batch_size, grid_h, grid_w, self.box_num, 4 + 1 + self.class_num))

            coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

            cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
            cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
            cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.box_num, 1])

            y_pred = tf.Print(y_pred, [tf.truediv(tf.reduce_sum(y_true[..., 4] - tf.sigmoid(y_pred[..., 4])*y_true[..., 4]), tf.reduce_sum(y_true[..., 4]))], message='confidence', summarize=1000)
            pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
            pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.box_num, 2])
            pred_box_conf = tf.sigmoid(y_pred[..., 4])
            pred_box_class = y_pred[..., 5:]

            pred_wh_half = pred_box_wh / 2.
            pred_mins = pred_box_xy - pred_wh_half
            pred_maxes = pred_box_xy + pred_wh_half

            true_box_xy = y_true[..., 0:2]
            true_box_wh = y_true[..., 2:4]
            true_wh_half = true_box_wh / 2.
            true_mins = true_box_xy - true_wh_half
            true_maxes = true_box_xy + true_wh_half

            intersect_mins = tf.maximum(pred_mins, true_mins)
            intersect_maxes = tf.minimum(pred_maxes, true_maxes)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

            true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
            pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

            union_areas = pred_areas + true_areas - intersect_areas
            iou_scores = tf.truediv(intersect_areas, union_areas)

            true_box_conf = iou_scores * y_true[..., 4]

            conf_mask = (1 - y_true[..., 4]) * self.no_object_scale
            conf_mask = conf_mask + y_true[..., 4] * self.object_scale

            true_box_class = tf.argmax(y_true[..., 5:], -1)
            class_mask = y_true[..., 4]

            nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
            nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
            nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

            loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
            loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
            loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
            loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
            loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
            loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
            loss_cc = tf.reduce_sum(true_box_conf)

            loss = loss_xy + loss_wh + loss_conf + loss_class

            loss = tf.Print(loss, [loss_xy], message="\nLoss Center Position")
            loss = tf.Print(loss, [loss_wh], message="Loss Width Height")
            loss = tf.Print(loss, [loss_conf], message="Loss Confidence")
            loss = tf.Print(loss, [loss_class], message="Loss Classification")
            loss = tf.Print(loss, [loss], message="Total Loss")

            return loss
        return yolo_loss

    def train(self):
        generator_config = {
            "IMAGE_H": self.image_height,
            "IMAGE_W": self.image_width,
            "GRID_H": self.grid_h,
            "GRID_W": self.grid_w,
            "BOX": self.box_num,
            "LABELS": self.labels,
            "CLASS": self.class_num,
            "ANCHORS": self.anchors,
            "BATCH_SIZE": self.batch_size,
            "MULTI_SCALE_TRAINING": self.multi_scale_training
        }

        if self.pretrained_weight is not None:
            if not os.path.exists(self.pretrained_weight):
                raise FileNotFoundError("{} is not exist".format(self.pretrained_weight))

            print("Load weights from {}".format(self.pretrained_weight))
            self.model.load_weights(self.pretrained_weight)

        if not os.path.exists(self.train_image_path) or not os.path.exists(self.train_annotation_path):
            raise FileNotFoundError("train dataset files are not found")
        if not os.path.exists(self.valid_image_path) or not os.path.exists(self.train_annotation_path):
            raise FileNotFoundError("valid dataset files are not found")

        weight_save_path = os.path.join(self.result_path, "weight")
        os.makedirs(weight_save_path, exist_ok=True)

        train_imgs = load_npy(self.train_image_path)
        train_annotations = load_npy(self.train_annotation_path)
        train_generator = BatchGenerator(train_imgs, train_annotations, generator_config, augmentation=True, shuffle=False)

        valid_imgs = load_npy(self.valid_image_path)
        valid_annotations = load_npy(self.valid_annotation_path)
        valid_generator = BatchGenerator(valid_imgs, valid_annotations, generator_config, augmentation=False, shuffle=False)

        checkpoint_train = ModelCheckpoint(filepath=os.path.join(weight_save_path, "weights_train.h5"),
                                           verbose=1,
                                           save_weights_only=True,
                                           mode="min",
                                           period=10)

        checkpoint_valid = ModelCheckpoint(filepath=os.path.join(weight_save_path, "weights_valid.h5"),
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode="min",
                                           period=1)

        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=self.lr_decay_rate,
                                      patience=5, min_lr=self.min_lr)

        print("Start Training...")
        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=self.epoch,
                                 verbose=1,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator),
                                 callbacks=[checkpoint_train, checkpoint_valid, reduce_lr],
                                 shuffle=False,
                                 workers=8,
                                 max_queue_size=50)
        print("End Training!")

    def inference(self, weight_path, image_path, obj_threshold, nms_threshold):
        if not os.path.exists(weight_path):
            raise FileNotFoundError("{} is not exist".format(weight_path))

        if not os.path.exists(image_path):
            raise FileNotFoundError("{} is not exist".format(image_path))

        print("Load weight from {}".format(weight_path))
        self.model.load_weights(weight_path)

        print("Start Inference...")

        image = load_image(image_path)

        self.prediction(image, obj_threshold, nms_threshold)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()

        print("End Inference!")

    def inference_valid(self, weight_path, save_path, obj_threshold, nms_threshold):
        if not os.path.exists(weight_path):
            raise FileNotFoundError("{} is not exist".format(weight_path))

        os.makedirs(save_path, exist_ok=True)

        print("Load weight from {}".format(weight_path))
        self.model.load_weights(weight_path)

        print("Load valid data...")
        valid_image = load_npy(self.valid_image_path)

        for i, image in enumerate(tqdm(valid_image, desc='Validation data prediction')):
            image = self.prediction(image, obj_threshold, nms_threshold)
            cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), image[:, :, ::-1])

        print('Result images are saved in {}'.format(save_path))
        print('End valid inference!')

    def prediction(self, image, obj_threshold, nms_threshold):
        input_image = cv2.resize(image, (self.image_height, self.image_width))
        input_image = input_image / 255.
        input_image = np.expand_dims(input_image, 0)

        dummy_input = np.zeros((1, 2))
        netout = self.model.predict([input_image, dummy_input])

        boxes = decode_netout(netout[0],
                              shape_dims=(self.grid_h, self.grid_w, self.box_num, 4 + 1 + self.class_num),
                              anchors=self.anchors,
                              nb_class=self.class_num,
                              obj_threshold=obj_threshold,
                              nms_threshold=nms_threshold
                              )

        image = draw_boxes(image, boxes, self.grid_h, self.grid_w, labels=self.labels)

        return image

    def mAP_evalutation(self, weight_path, iou_threshold, obj_threshold, nms_threshold):
        '''
        Reference this blog
        https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge
        '''
        if not os.path.exists(weight_path):
            raise FileNotFoundError('{} is not exists'.format(weight_path))

        generator_config = {
            "IMAGE_H": self.image_height,
            "IMAGE_W": self.image_width,
            "GRID_H": self.grid_h,
            "GRID_W": self.grid_w,
            "BOX": self.box_num,
            "LABELS": self.labels,
            "CLASS": self.class_num,
            "ANCHORS": self.anchors,
            "BATCH_SIZE": self.batch_size,
        }

        valid_imgs = load_npy(self.valid_image_path)
        valid_annotations = load_npy(self.valid_annotation_path)
        valid_generator = BatchGenerator(valid_imgs, valid_annotations, generator_config, augmentation=False, shuffle=False)
        generator = valid_generator

        print('Load pretrained weight {}...'.format(weight_path))
        self.model.load_weights(weight_path)

        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        dummy_input = np.zeros((1, 2))

        for i in tqdm(range(generator.size()), desc='mAP evaluation'):
            image = generator.load_image(i)
            height, width, channels = image.shape

            input_image = cv2.resize(image, (self.image_height, self.image_width))
            input_image = input_image / 255.
            input_image = np.expand_dims(input_image, 0)

            netout = self.model.predict([input_image, dummy_input])
            pred_boxes = decode_netout(netout[0], (self.grid_h, self.grid_w, self.box_num, 4 + 1 + self.class_num), self.anchors, self.class_num, obj_threshold, nms_threshold)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*width/self.grid_w, box.ymin*height/self.grid_h, box.xmax*width/self.grid_w, box.ymax*height/self.grid_h, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        average_precisions = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0
            detected_box = 0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []
                detected_box += len(detections)

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps)
                    max_overlap = overlaps[assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            if num_annotations == 0:
                if detected_box != 0:
                    average_precisions[label] = 0
                continue

            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            recall = true_positives / num_annotations
            precision = true_positives / (true_positives + false_positives)

            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    def show_me_camera(self, weight_path, obj_threshold, nms_threshold):
        if not os.path.exists(weight_path):
            raise FileNotFoundError('{} is not exists'.format(weight_path))

        print('Load weight from {}'.format(weight_path))
        self.model.load_weights(weight_path)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise NotImplemented('Unable to read camera feed')

        print('Video capture start!')

        while True:
            start = time.time()
            ret, frame = cap.read()

            if ret:
                input_image = cv2.resize(frame, (self.image_height, self.image_width))
                input_image = input_image / 255.
                input_image = np.expand_dims(input_image, 0)

                dummy_input = np.zeros((1, 2))
                netout = self.model.predict([input_image, dummy_input])

                boxes = decode_netout(netout[0],
                                      shape_dims=(self.grid_h, self.grid_w, self.box_num, 4 + 1 + self.class_num),
                                      obj_threshold=obj_threshold,
                                      nms_threshold=nms_threshold,
                                      anchors=self.anchors,
                                      nb_class=self.class_num)

                frame = draw_boxes(frame, boxes, self.grid_h, self.grid_w, labels=self.labels)
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

    def __set_variable(self, key, default_value, config):
        return config[key] if key in config.keys() else default_value
