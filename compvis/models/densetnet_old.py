import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

weight_decay = 1e-4

def conv_block(x, growth_rate, dropout_rate=0.2):
    x1 = layers.BatchNormalization()(x)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv1D(4 * growth_rate, kernel_size=1, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x1)
    x1 = layers.Dropout(dropout_rate)(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv1D(growth_rate, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x1)
    x1 = layers.Dropout(dropout_rate)(x1)

    x = layers.Concatenate()([x, x1])
    return x

def dense_block(x, num_convs, growth_rate, dropout_rate=0.2):
    for _ in range(num_convs):
        x = conv_block(x, growth_rate, dropout_rate)
    return x

def transition_block(x, reduction, dropout_rate=0.2):
    num_filters = int(x.shape[-1] * reduction)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(num_filters, kernel_size=1, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.AveragePooling1D(2, strides=2)(x)
    return x

def create_densenet169(input_shape, num_classes, growth_rate=32, compression=0.5, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Dense Blocks and Transition Layers
    num_blocks = [6, 12, 32, 32]  # The number of layers in each block for DenseNet169

    for i, num_layers in enumerate(num_blocks):
        x = dense_block(x, num_layers, growth_rate, dropout_rate)
        if i != len(num_blocks) - 1:  # no transition block after the last dense block
            x = transition_block(x, compression, dropout_rate)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model