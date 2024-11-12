# vision_transformer.py

import tensorflow as tf

from compvis.embedding.patch_embedding import PatchEmbedding
from compvis.layers.transformer_encoder import TransformerEncoderLayer


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


if __name__ == "__main__":
    # Sample input dimensions and model parameters
    batch_size = 2
    img_size = 32  # Example image size (32x32)
    patch_dim = 4
    embed_dim = 64
    num_layers = 2
    num_heads = 4
    mlp_units = 128
    num_classes = 10  # Example number of classes
    drop_rate = 0.1

    # Create a sample input tensor and a dummy structural feature input
    sample_input = tf.random.normal((batch_size, img_size, img_size, 3))  # RGB image
    struct_feats = tf.random.normal((batch_size, 5))  # Example structural features

    # Instantiate the Vision Transformer model
    vit_model = VisionTransformerModel(
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_units=mlp_units,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Perform a forward pass
    output = vit_model((sample_input, struct_feats), training=False)

    # Print the model summary and the output shape
    vit_model.summary()
    print("Output shape:", output.shape)
