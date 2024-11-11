# vision_transformer.py

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


class VisionTransformerModel(tf.keras.Model):
    def __init__(
        self,
        patch_dim,
        embed_dim,
        num_layers,
        num_heads,
        mlp_units,
        num_classes,
        drop_rate=0.0,
        survival_prob=1.0,
        use_attn_bias=False,
        attn_dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_layer = PatchEmbedding(patch_dim, embed_dim, drop_rate)
        self.encoder_layers = [
            TransformerEncoderLayer(
                num_heads,
                embed_dim,
                use_attn_bias,
                mlp_units,
                attn_dropout_rate,
                survival_prob if i == 0 else 1.0,
                dropout_rate=drop_rate,
                film_layer=FiLMLayer(),
            )
            for i in range(num_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x, struct_feats = inputs
        x = self.embedding_layer(x, training=training)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer([x, struct_feats], training=training)
        x = self.final_norm(x)
        return self.classifier(x[:, 0])

    def get_final_attention(self, x):
        x = self.embedding_layer(x, training=False)
        for encoder_layer in self.encoder_layers[:-1]:
            x = encoder_layer([x, None], training=False)
        return self.encoder_layers[-1].get_attention_weights(x)
