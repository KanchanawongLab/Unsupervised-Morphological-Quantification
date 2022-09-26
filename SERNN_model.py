import os
import glob
import numpy as np

#tf.keras.models.Model does not allow for model.save for some strange reasons
import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import selu
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

class seq_model(object):
    def __init__(self):
        print("Using SEQUENTIAL model")

    def conv_model(self, no_of_classes):
        kernel_size = (3,3) #A slightly bigger kernel size to capture more details at once
        pad = 'same'
        filters = 16
        initial_input = Input(shape = (256, 256, 3))

        def se_block(input_layer, filters, ratio = 8):
            #Squeeze and Excitation network
            #To recalibrate the feature maps
            p1 = GlobalAveragePooling2D()(input_layer)
            d1 = Dense(filters//ratio)(p1)
            a1 = ELU()(d1)
            d2 = Dense(filters, activation = 'sigmoid')(a1)
            scale = multiply([input_layer, d2])

            return scale

        def residual_block(input_layer, filters, kernel_size = kernel_size, padding = pad):
            #Obtain the number of filters in the input layer
            #This is to check whether the number of output filters is equal to the number of input filters to avoid issues
            initial_filters = input_layer.get_shape().as_list()[3]
            if filters != initial_filters:
                skip_layer = Conv2D(filters, [1, 1], strides = [1, 1], padding = pad)(input_layer)
            else:
                skip_layer = input_layer

            #Create the layers
            c1 = Conv2D(filters, kernel_size, padding = pad, kernel_initializer = "he_normal")(skip_layer)
            b1 = BatchNormalization()(c1)
            a1 = ELU()(b1)

            c2 = Conv2D(filters, kernel_size, padding = pad, kernel_initializer = "he_normal")(a1)
            b2 = BatchNormalization()(c2)

            se_output = se_block(b2, filters)

            output = Add()([skip_layer, se_output])
            output = ELU()(output)

            return output

        block_1 = residual_block(initial_input, filters, kernel_size = (7,7))
        p1 = AveragePooling2D()(block_1) #(256, 256, 16)

        block_2 = residual_block(p1, filters*2)
        block_2 = residual_block(block_2, filters*2)
        block_2 = residual_block(block_2, filters*2)
        p2 = AveragePooling2D()(block_2) #(128, 128, 32)

        block_3 = residual_block(p2, filters*2)
        block_3 = residual_block(block_3, filters*2)
        block_3 = residual_block(block_3, filters*2)
        p3 = AveragePooling2D()(block_3) #(64, 64, 32)

        block_4 = residual_block(p3, filters*4)
        block_4 = residual_block(block_4, filters*4)
        block_4 = residual_block(block_4, filters*4)
        p4 = AveragePooling2D()(block_4) #(16, 16, 64)

        block_5 = residual_block(p4, filters*4)
        block_5 = residual_block(block_5, filters*4)
        block_5 = residual_block(block_5, filters*4)  
        p5 = AveragePooling2D()(block_5) #(8, 8, 64)

        block_6 = residual_block(p5, filters*8)
        block_6 = residual_block(block_6, filters*8)
        block_6 = residual_block(block_6, filters*8)
        p6 = AveragePooling2D()(block_6) #(4, 4, 128)

        g = GlobalAveragePooling2D()(p6) #128
        r = Dropout(0.3)(g) #To reduce overfitting

        output = Dense(no_of_classes, activation = 'softmax')(r)

        model = Model(inputs = [initial_input], outputs = [output])

        return model

#class_obtain = seq_model()
#model = class_obtain.conv_model(12) #6 ECM proteins x 2 cell types = 12 classes

#print(model.summary())


