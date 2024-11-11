import tensorflow as tf

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        head_count,
        key_dim,
        use_attn_bias,
        mlp_units,
        attn_dropout_rate=0.0,
        survival_prob=1.0,
        activation_function="gelu",
        dropout_rate=0.0,
        film_layer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(
            head_count,
            key_dim // head_count,
            use_bias=use_attn_bias,
            dropout=attn_dropout_rate,
        )
        self.stochastic_depth_layer = tfa.layers.StochasticDepth(survival_prob)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ffn = FeedForwardNetwork(
            hidden_units=mlp_units,
            activation_function=activation_function,
            drop_rate=dropout_rate,
        )
        self.film_layer = film_layer

    def call(self, inputs, training=False):
        x, struct_feats = inputs
        norm_x = self.norm1(x, training=training)
        attn_output = self.multi_head_attn(norm_x, norm_x, training=training)
        x = self.stochastic_depth_layer([x, attn_output], training=training)
        if self.film_layer:
            x = self.film_layer([x, struct_feats])
        ffn_output = self.ffn(self.norm2(x, training=training), training=training)
        x = self.stochastic_depth_layer([x, ffn_output], training=training)
        return x

    def get_attention_weights(self, x):
        norm_x = self.norm1(x, training=False)
        return self.multi_head_attn(
            norm_x, norm_x, training=False, return_attention_scores=True
        )[1]