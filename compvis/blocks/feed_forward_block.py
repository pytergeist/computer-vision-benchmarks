import tensorflow as tf

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units,
        output_units=None,
        activation_function="gelu",
        drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.activation_function = activation_function
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.dense_layer1 = tf.keras.layers.Dense(self.hidden_units)
        self.activation_layer = tf.keras.layers.Activation(self.activation_function)
        self.dropout_layer = tf.keras.layers.Dropout(self.drop_rate)
        self.dense_layer2 = tf.keras.layers.Dense(
            input_shape[-1] if self.output_units is None else self.output_units
        )

    def call(self, x, training=False):
        x = self.dense_layer1(x)
        x = self.activation_layer(x)
        x = self.dropout_layer(x, training=training)
        x = self.dense_layer2(x)
        x = self.dropout_layer(x, training=training)
        return x