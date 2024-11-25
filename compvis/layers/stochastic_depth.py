# stochastic_depth.py

import tensorflow as tf


class StochasticDepth(tf.keras.layer.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        pass
