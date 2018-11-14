from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input, Reshape, LeakyReLU
from keras.models import Model

'''
모델만 구현해놓음
학습 코드는 v2에 대해서만 구현해놓음
'''


def yolo(input_shape=(448, 448, 3), model_name='yolo_v1'):

    def Conv2D_LeackyReLU(filters, kernel_size, strides=1, padding='same'):
        def _Conv2D_LeackyReLU(x):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
            x = LeakyReLU(alpha=0.1)(x)
            return x

        return _Conv2D_LeackyReLU

    inp = Input(shape=input_shape)

    conv_1_1 = Conv2D_LeackyReLU(filters=64, kernel_size=7, strides=2)(inp) # 224
    maxpool_1_1 = MaxPooling2D(pool_size=2)(conv_1_1) # 112

    conv_2_1 = Conv2D_LeackyReLU(filters=192, kernel_size=3)(maxpool_1_1)
    maxpool_2_1 = MaxPooling2D(pool_size=2)(conv_2_1) # 56

    conv_3_1 = Conv2D_LeackyReLU(filters=128, kernel_size=1)(maxpool_2_1)
    conv_3_2 = Conv2D_LeackyReLU(filters=256, kernel_size=3)(conv_3_1)
    conv_3_3 = Conv2D_LeackyReLU(filters=256, kernel_size=1)(conv_3_2)
    conv_3_4 = Conv2D_LeackyReLU(filters=512, kernel_size=3)(conv_3_3)
    maxpool_3_1 = MaxPooling2D(pool_size=2)(conv_3_4) # 28

    conv_4 = maxpool_3_1

    for i in range(4):
        conv_4 = Conv2D_LeackyReLU(filters=256, kernel_size=1)(conv_4)
        conv_4 = Conv2D_LeackyReLU(filters=512, kernel_size=3)(conv_4)

    conv_4_1 = Conv2D_LeackyReLU(filters=512, kernel_size=1)(conv_4)
    conv_4_2 = Conv2D_LeackyReLU(filters=1024, kernel_size=3)(conv_4_1)
    maxpool_4 = MaxPooling2D(pool_size=2)(conv_4_2) # 14

    conv_5 = maxpool_4

    for i in range(2):
        conv_5 = Conv2D_LeackyReLU(filters=512, kernel_size=1)(conv_5)
        conv_5 = Conv2D_LeackyReLU(filters=1024, kernel_size=3)(conv_5)

    conv_5_1 = Conv2D_LeackyReLU(filters=1024, kernel_size=3)(conv_5)
    conv_5_2 = Conv2D_LeackyReLU(filters=1024, kernel_size=3, strides=2)(conv_5_1) # 7

    conv_6_1 = Conv2D_LeackyReLU(filters=1024, kernel_size=3)(conv_5_2)
    conv_6_2 = Conv2D_LeackyReLU(filters=1024, kernel_size=3)(conv_6_1)

    fc_1 = LeakyReLU(alpha=0.1)(Dense(units=4096)(Flatten()(conv_6_2)))
    fc_2 = Dense(units=7*7*30, activation='linear')(fc_1)

    yolo_output = Reshape(target_shape=(-1, 7, 7, 30))(fc_2)

    model = Model(inputs=inp, outputs=yolo_output, name=model_name)
    model.summary()

    return model