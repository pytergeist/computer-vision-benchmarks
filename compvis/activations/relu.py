# relu.py

import tensorflow as tf


class ReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def call(self, inputs):
        self._validate_input_populated(inputs)
        self._validate_input_is_tensor(inputs)
        return tf.math.maximum(inputs, 0)

    @staticmethod
    def _validate_input_populated(inputs):
        if inputs is None:
            raise ValueError("Input to ReLU cannot be None.")

    @staticmethod
    def _validate_input_is_tensor(inputs):
        if not isinstance(inputs, tf.Tensor):
            raise TypeError(
                f"Input must be a TensorFlow tensor, but got {type(inputs)}"
            )

    def get_config(self):
        config = super(ReLU, self).get_config()
        return config


if __name__ == "__main__":
    inputs = tf.random.uniform(shape=[1, 10, 10], minval=-0.5, maxval=0.9)
    relu = ReLU()
    outputs = relu(inputs)
    print("Inputs:\n", inputs)
    print("Outputs:\n", outputs)
