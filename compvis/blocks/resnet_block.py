# resnet_block.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(
        self, filters, kernel_size, strides, weight_decay, dropout_rate=None, **kwargs
    ):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn2 = layers.BatchNormalization()

        self.shortcut_conv = None
        self.shortcut_bn = None

    def build_shortcut(self, inputs):
        if self.shortcut_conv is None and inputs.shape[-1] != self.conv1.filters:
            self.shortcut_conv = layers.Conv1D(
                filters=self.conv1.filters,
                kernel_size=1,
                strides=self.conv1.strides,
                padding="same",
                kernel_regularizer=regularizers.l2(self.conv1.kernel_regularizer.l2),
            )
            self.shortcut_bn = layers.BatchNormalization()

    def apply_shortcut(self, inputs, training):
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        return shortcut

    def residual_path(self, inputs, training):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return x

    def call(self, inputs, training=False):
        self.build_shortcut(inputs)
        shortcut = self.apply_shortcut(inputs, training)

        x = self.residual_path(inputs, training)

        x = layers.Add()([shortcut, x])
        x = self.relu(x)

        return x
