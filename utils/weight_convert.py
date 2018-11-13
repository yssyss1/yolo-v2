import sys
sys.path.append("..")

import numpy as np
from model.yolo import yolo
import os
import baker


class WeightConverter:
    def __init__(self, weight_file_path):
        self.offset = 4
        if not os.path.exists(weight_file_path):
            raise FileNotFoundError("{} path is not exist".format(weight_file_path))
        self.all_weights = np.fromfile(weight_file_path, dtype="float32")

        self.model = self.build_model()
        self.nb_conv = 23

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4

    def build_model(self):
        return yolo(image_width=416, image_height=416, grid_w=13, grid_h=13, class_num=80, box_num=5)

    def weight_convert_h5(self, dest_path):
        self.reset()

        print("Start Converting Weights to h5 file...")

        outname = "yolo"
        folder_path = dest_path

        path = os.path.splitext(dest_path)

        if path[-1] != "":
            folder_path = os.path.split(dest_path)[0]
            outname = path[0].split("/")[-1]

        os.makedirs(folder_path, exist_ok=True)

        for i in range(1, self.nb_conv + 1):
            conv_layer = self.model.get_layer("conv_{}".format(i))

            if i < self.nb_conv:
                norm_layer = self.model.get_layer("norm_{}".format(i))

                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = self.read_bytes(size)
                gamma = self.read_bytes(size)
                mean = self.read_bytes(size)
                var = self.read_bytes(size)

                norm_layer.set_weights([gamma, beta, mean, var])

            if len(conv_layer.get_weights()) > 1:
                bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])

        print("End Converting Weights to h5 file...")

        self.model.save_weights(filepath="{}.h5".format(os.path.join(folder_path, outname)))


@baker.command(
    params={
        "weight_file_path": "yolov2.weights 경로",
        "dest_path": "keras weight 저장 경로",
    }
)
def convert_yolo_weight_keras(weight_file_path, dest_path):
    weight_converter = WeightConverter(weight_file_path=weight_file_path)
    weight_converter.weight_convert_h5(dest_path=dest_path)


if __name__ == "__main__":
    baker.run()
