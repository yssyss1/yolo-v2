from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import tensorflow as tf


def yolo(image_width=416, image_height=416, grid_w=13, grid_h=13, class_num=80, box_num=5):

    def conv_block(filters, kernel_size, strides, idx, padding='same', use_bias=False, use_batchnorm=True, use_maxpool=True, pool_size=(2, 2), alpha=0.1):
        def _conv_block(x):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name='conv_{}'.format(idx), use_bias=use_bias)(x)
            x = BatchNormalization(name='norm_{}'.format(idx))(x) if use_batchnorm else x
            x = LeakyReLU(alpha=alpha)(x)
            x = MaxPooling2D(pool_size=pool_size)(x) if use_maxpool else x

            return x

        return _conv_block

    input_image = Input(shape=(image_height, image_width, 3))

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

    skip_connection = conv_block(filters=64, kernel_size=(1, 1), strides=(1, 1), idx=21, use_maxpool=False)(skip_connection)
    skip_connection = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(skip_connection)

    x = concatenate([skip_connection, x])

    x = conv_block(filters=1024, kernel_size=(3, 3), strides=(1, 1), idx=22, use_maxpool=False)(x)
    x = Conv2D(box_num * (4 + 1 + class_num), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((grid_h, grid_w, box_num, 4 + 1 + class_num))(x)

    model = Model(input_image, output)
    model.summary()

    return model