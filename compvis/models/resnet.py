# resnet.py

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

from compvis.blocks.resnet_block import ResNetBlock


class ResNet(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        num_filters=64,
        num_blocks=[2, 2, 2, 2],
        dropout_rate=0.2,
        **kwargs
    ):
        super(ResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.initial_conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_relu = layers.ReLU()
        self.initial_pool = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")

        self.resnet_blocks = []
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                strides = 2 if i > 0 and j == 0 else 1
                self.resnet_blocks.append(
                    ResNetBlock(
                        filters=num_filters * (2**i),
                        kernel_size=3,
                        strides=strides,
                        dropout_rate=dropout_rate,
                    )
                )

        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(units=num_classes, activation="relu")

    def generate_resnet_blocks(self):
        self.resnet_blocks = []
        for i, num_block in enumerate(self.num_blocks):
            for j in range(num_block):
                strides = 2 if i > 0 and j == 0 else 1
                self.resnet_blocks.append(
                    ResNetBlock(
                        filters=self.num_filters * (2 ** i),
                        kernel_size=3,
                        strides=strides,
                        dropout_rate=self.dropout_rate,
                    )
                )

    def call(self, inputs, training=False):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)
        x = self.initial_pool(x)

        for block in self.resnet_blocks:
            x = block(x, training=training)

        x = self.global_avg_pool(x)
        x_out = self.fc(x)

        return x_out
