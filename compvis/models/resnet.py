# resnet.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from compvis.blocks.resnet_block import ResNetBlock


class ResNet(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        num_filters=64,
        num_blocks=[2, 2, 2, 2],
        weight_decay=1e-4,
        dropout_rate=0.2,
        **kwargs
    ):
        super(ResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        self.initial_conv = layers.Conv2D(
            filters=num_filters,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_relu = layers.ReLU()
        self.initial_pool = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")

        self.resnet_blocks = self.generate_resnet_blocks()

        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(units=num_classes, activation="relu")

    def generate_resnet_blocks(self):
        blocks = []
        for i, num_block in enumerate(self.num_blocks):
            for j in range(num_block):
                strides = 2 if i > 0 and j == 0 else 1
                blocks.append(
                    ResNetBlock(
                        filters=self.num_filters * (2**i),
                        kernel_size=3,
                        strides=strides,
                        dropout_rate=self.dropout_rate,
                        weight_decay=self.weight_decay,
                    )
                )
        return blocks

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


if __name__ == "__main__":
    import numpy as np

    input_shape = (32, 128, 64)
    num_classes = 10
    num_filters = 64
    num_blocks = [2, 2, 2, 2]
    weight_decay = 1e-4

    test_input = np.random.randn(*input_shape).astype(np.float32)

    model = ResNet(
        num_classes=num_classes,
        num_filters=num_filters,
        num_blocks=num_blocks,
        weight_decay=weight_decay,
    )

    model.build(input_shape=(None, input_shape[1], input_shape[2]))

    outputs = model(test_input, training=False)
    print("Model output shape:", outputs.shape)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    test_labels = np.random.randint(0, num_classes, size=(input_shape[0],))

    history = model.fit(test_input, test_labels, epochs=1, verbose=1)

    model.summary()
