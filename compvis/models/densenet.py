# densenet.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from compvis.blocks.densenet_block import DenseNetBlock


class DenseNet(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        num_blocks=3,
        growth_rate=12,
        bottleneck_size=4,
        compression_factor=0.5,
        num_filters=64,
        weight_decay=1e-4,
        dropout_rate=0.2,
        **kwargs
    ):
        super(DenseNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.bottleneck_size = bottleneck_size
        self.compression_factor = compression_factor
        self.num_filters = num_filters
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Initial Conv layer
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

        self.densenet_blocks = self.generate_densenet_blocks()

        self.compression_conv = layers.Conv2D(
            filters=int(self.num_filters * self.compression_factor),
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.compression_bn = layers.BatchNormalization()

        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(units=num_classes, activation="softmax")

    def generate_densenet_blocks(self):
        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(
                DenseNetBlock(
                    growth_rate=self.growth_rate,
                    bottleneck_size=self.bottleneck_size * self.growth_rate,
                    weight_decay=self.weight_decay,
                    dropout_rate=self.dropout_rate,
                )
            )
            # After each Dense Block, apply compression and transition to the next block
            blocks.append(layers.BatchNormalization())
            blocks.append(layers.ReLU())
            blocks.append(
                layers.Conv2D(
                    filters=int(self.num_filters * self.compression_factor),
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_regularizer=regularizers.l2(self.weight_decay),
                )
            )
            blocks.append(
                layers.AveragePooling1D(pool_size=2, strides=2, padding="same")
            )
        return blocks

    def call(self, inputs, training=False):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)
        x = self.initial_pool(x)

        for block in self.densenet_blocks:
            x = block(x, training=training)

        x = self.global_avg_pool(x)
        x_out = self.fc(x)

        return x_out


if __name__ == "__main__":
    import numpy as np

    input_shape = (32, 128, 64)
    num_classes = 10
    num_blocks = 3
    growth_rate = 12
    bottleneck_size = 4
    compression_factor = 0.5

    test_input = np.random.randn(*input_shape).astype(np.float32)

    model = DenseNet(
        num_classes=num_classes,
        num_blocks=num_blocks,
        growth_rate=growth_rate,
        bottleneck_size=bottleneck_size,
        compression_factor=compression_factor,
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
