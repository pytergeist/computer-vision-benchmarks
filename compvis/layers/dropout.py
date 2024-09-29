# dropout.py

import tensorflow as tf


class Dropout(tf.keras.layers.Layer):
    def __init__(self, rate, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            uniform_dist = tf.random.uniform(shape=tf.shape(inputs), maxval=1)
            dropout_mask = tf.cast(uniform_dist >= self.rate, dtype=tf.float32)
            return inputs * dropout_mask / (1 - self.rate)
        return inputs

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate):
        self._validate_rate(rate)
        self._rate = rate

    @staticmethod
    def _validate_rate(rate):
        assert 0 <= rate <= 1, "Dropout rate must be between 0 and 1"

    def get_config(self):
        config = super(Dropout, self).get_config()
        config.update(
            {
                "rate": self._rate,
            }
        )
        return config


if __name__ == "__main__":
    dropout = Dropout(rate=15)
    print(dropout.rate)
    print(dropout.get_config())
