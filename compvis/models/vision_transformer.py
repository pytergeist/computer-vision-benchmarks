# vision_transformer.py

import tensorflow as tf

from compvis.layers.transformer_encoder import TransformerEncoderLayer
from compvis.embedding.patch_embedding import PatchEmbedding

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
