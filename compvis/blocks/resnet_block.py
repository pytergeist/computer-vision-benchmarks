# resnet_block.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class ResNetBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, weight_decay, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay))
        self.bn2 = layers.BatchNormalization()

        self.shortcut_conv = None
        self.shortcut_bn = None
        if filters != self.conv1.input_shape[-1] or strides != 1:
            self.shortcut_conv = layers.Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same',
                                               kernel_regularizer=regularizers.l2(weight_decay))
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = layers.Add()([shortcut, x])
        x = self.relu(x)

        return x
