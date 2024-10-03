# resnet_block.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

weight_decay = 1e-4


def resnet_block(inputs, filters, kernel_size, strides, dropout_rate):
    shortcut = inputs
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = layers.BatchNormalization()(x)

    if filters != inputs.shape[-1] or strides != 1:
        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )(inputs)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)

    return x


def create_hybrid_resnet(
    rdf_shape, num_classes, num_filters=64, num_blocks=[2, 2, 2, 2], dropout_rate=0.2
):
    rdf_input = layers.Input(shape=rdf_shape)

    x = layers.Conv1D(
        filters=num_filters,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(rdf_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    for i, num_block in enumerate(num_blocks):
        for j in range(num_block):
            strides = 2 if i > 0 and j == 0 else 1

            x = resnet_block(
                x,
                filters=num_filters * (2**i),
                kernel_size=3,
                strides=strides,
                dropout_rate=dropout_rate,
            )

    x = layers.GlobalAveragePooling1D()(x)
    x_out = layers.Dense(units=num_classes, activation="relu")(x)

    model = tf.keras.Model(inputs=rdf_input, outputs=x_out)

    return model
