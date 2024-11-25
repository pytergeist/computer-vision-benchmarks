import tensorflow as tf


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_dim, embed_dim, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.conv_proj = tf.keras.layers.Conv2D(
            filters=embed_dim, kernel_size=patch_dim, strides=patch_dim
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=drop_rate)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim), trainable=True, name="cls_token"
        )
        num_patches = (input_shape[1] // self.patch_dim) * (
            input_shape[2] // self.patch_dim
        )
        self.pos_embedding = self.add_weight(
            shape=(1, num_patches + 1, self.embed_dim),
            trainable=True,
            name="pos_embedding",
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        x = self.conv_proj(inputs)
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding
        x = self.dropout_layer(x, training=training)
        return x
