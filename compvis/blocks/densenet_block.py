# densenet_block.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers


class DenseNetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        growth_rate,
        bottleneck_size,
        weight_decay=1e-4,
        dropout_rate=None,
        **kwargs
    ):
        super(DenseNetBlock, self).__init__(**kwargs)
        self.growth_rate = growth_rate
        self.bottleneck_size = bottleneck_size
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self.bottleneck_conv = layers.Conv2D(
            filters=bottleneck_size,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.dropout = layers.Dropout(dropout_rate) if dropout_rate else None

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = self.relu1(x)
        x = self.bottleneck_conv(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return tf.concat([inputs, x], axis=-1)
