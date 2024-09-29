# relu.py

import tensorflow as tf


class ReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.maximum(inputs, 0)

    def get_config(self):
        config = super(ReLU, self).get_config()
        return config


if __name__ == "__main__":
    inputs = tf.random.uniform(shape=[1, 10, 10], minval=-0.5, maxval=0.9)
    relu = ReLU()
    outputs = relu(inputs)
    print(inputs)
    print(outputs)
