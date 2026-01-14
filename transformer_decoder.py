"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains classes and functions for Transformer Decoder.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    LayerNormalization,
    Dense,
    Dropout,
    Layer,
)
from pixel_decoder import SinePositionEmbedding


class MaskedMultiHeadAttention(Layer):
    """
    Custom Multi-Head Attention Layer for Mask2Former.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        **kwargs: Additional keyword arguments for the base Layer class.

    Raises:
        ValueError: If embed_dim is not divisible by num_heads.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
        super(MaskedMultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Layers
        self.q_proj = Dense(embed_dim)
        self.k_proj = Dense(embed_dim)
        self.v_proj = Dense(embed_dim)
        self.out_proj = Dense(embed_dim)

        self.dropout_layer = Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """
        Splits last dim: (B, Seq, Embed) -> (B, Heads, Seq, HeadDim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, attention_mask=None, training=False):
        """
        Applies masked multi-head attention.

        Args:
            query (tf.Tensor): Query tensor of shape [B, N_q, embed_dim].
            key (tf.Tensor): Key tensor of shape [B, N_k, embed_dim].
            value (tf.Tensor): Value tensor of shape [B, N_k, embed_dim].
            attention_mask (tf.Tensor, optional): Additive attention mask. Defaults to None.
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor of shape [B, N_q, embed_dim].
        """
        batch_size = tf.shape(query)[0]

        # Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Split heads: (B, H, N_q, D_h)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product: (B, H, N_q, N_k)
        attn_logits = tf.matmul(q, k, transpose_b=True)
        attn_logits = attn_logits * self.scale

        # Apply mask bias
        if attention_mask is not None:
            attn_logits += attention_mask

        # Softmax
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)

        # Dropout
        if training:
            attn_weights = self.dropout_layer(attn_weights)

        # Weighted sum: (B, H, N_q, D_h)
        output = tf.matmul(attn_weights, v)

        # Concatenate heads: (B, N_q, E)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))

        # Output projection
        output = self.out_proj(output)

        return output

# Transformer decoder building blocks
class TransformerDecoderLayer(Layer):
    """
    A single Transformer decoder layer:
    - self-attention on queries
    - cross-attention from queries to memory (pixel features)
    - feed-forward network

    Args:
        d_model (int): Model dimension. Defaults to 256.
        num_heads (int): Number of attention heads. Defaults to 8.
        dim_feedforward (int): Dimension of feed-forward network. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.self_attn = MaskedMultiHeadAttention(
            num_heads=num_heads, embed_dim=d_model, dropout_rate=dropout,
            name=self.name + "_self_attn"
        )
        self.cross_attn = MaskedMultiHeadAttention(
            num_heads=num_heads, embed_dim=d_model, dropout_rate=dropout,
            name=self.name + "_cross_attn"
        )

        self.norm1 = LayerNormalization(epsilon=1e-5, name=self.name + "_norm1")
        self.norm2 = LayerNormalization(epsilon=1e-5, name=self.name + "_norm2")
        self.norm3 = LayerNormalization(epsilon=1e-5, name=self.name + "_norm3")

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.linear1 = Dense(dim_feedforward, activation="relu", name=self.name + "_ffn1")
        self.linear2 = Dense(d_model, name=self.name + "_ffn2")

    def call(
            self,
            tgt,  # [B, Q, C]
            memory,  # [B, S, C]
            query_pos=None,  # [B, Q, C] or None
            memory_pos=None,  # [B, S, C] or None
            training=False,
            cross_attention_mask=None
    ):
        """
        Applies transformer decoder layer operations.

        Args:
            tgt (tf.Tensor): Target query embeddings of shape [B, Q, C].
            memory (tf.Tensor): Encoder memory of shape [B, S, C].
            query_pos (tf.Tensor, optional): Query positional embeddings of shape [B, Q, C]. Defaults to None.
            memory_pos (tf.Tensor, optional): Memory positional embeddings of shape [B, S, C]. Defaults to None.
            training (bool): Whether in training mode. Defaults to False.
            cross_attention_mask (tf.Tensor, optional): Cross-attention mask. Defaults to None.

        Returns:
            tf.Tensor: Transformed query embeddings of shape [B, Q, C].
        """
        # Apply norm first
        tgt_norm = self.norm1(tgt)

        # Cross-attention with positional encodings
        if query_pos is not None:
            q = tgt_norm + query_pos
        else:
            q = tgt_norm

        if memory_pos is not None:
            k = memory + memory_pos
            v = memory
        else:
            k = v = memory

        # Masked cross-attention
        tgt2 = self.cross_attn(
            query=q,
            value=v,
            key=k,
            attention_mask=cross_attention_mask,
            training=training,
        )
        tgt = tgt + self.dropout1(tgt2, training=training)

        # Self-attention among queries
        tgt_norm = self.norm2(tgt)

        if query_pos is not None:
            q = tgt_norm + query_pos
            k = tgt_norm + query_pos
        else:
            q = k = tgt_norm

        tgt2 = self.self_attn(query=q, key=k, value=tgt_norm, training=training)
        tgt = tgt + self.dropout2(tgt2, training=training)

        # Feed-forward network
        tgt_norm = self.norm3(tgt)
        ff_output  = self.linear2(self.linear1(tgt_norm, training=training), training=training)
        tgt = tgt + self.dropout3(ff_output, training=training)

        return tgt

class TransformerDecoder(Layer):
    """
    Transformer Decoder with Masked Cross-Attention (Mask2Former-style),
    using multi-scale memories in round-robin scheduling.

    Args:
        num_layers (int): Number of decoder layers. Defaults to 6.
        d_model (int): Model dimension. Defaults to 256.
        num_heads (int): Number of attention heads. Defaults to 8.
        dim_feedforward (int): Dimension of feed-forward network. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the base Layer class.
    """
    def __init__(self, num_layers=6, d_model=256, num_heads=8,
                 dim_feedforward=1024, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            TransformerDecoderLayer(
                d_model=d_model, num_heads=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
                name=f"{self.name}_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.num_layers = num_layers

    def _resize_mask_logits_to_level(self, mask_logits, target_h, target_w):
        """Resize mask logits `mask_logits` to the resolution (target_h, target_w)."""
        # Reshape and resize mask logits
        B = tf.shape(mask_logits)[0]
        Q = tf.shape(mask_logits)[1]
        H_src = tf.shape(mask_logits)[2]
        W_src = tf.shape(mask_logits)[3]
        mask_logits_reshaped = tf.reshape(mask_logits, (B * Q, H_src, W_src, 1))
        mask_logits_resized = tf.image.resize(mask_logits_reshaped, size=(target_h, target_w), method='bilinear')
        mask_logits_resized = tf.reshape(mask_logits_resized, (B, Q, target_h, target_w))
        return mask_logits_resized

    def _create_attention_bias(self, mask_logits, target_h, target_w):
        """
        Creates an additive attention bias from mask logits.
        Returns tensor of shape [B, Q, S] with values 0.0 or -1e9.
        """
        # Resize and threshold logits
        resized_logits = self._resize_mask_logits_to_level(mask_logits, target_h, target_w)
        mask_bool = tf.stop_gradient(resized_logits > 0.0)  # [B, Q, H, W]

        # Flatten and create additive bias
        B = tf.shape(mask_bool)[0]
        Q = tf.shape(mask_bool)[1]
        mask_bool_flat = tf.reshape(mask_bool, [B, Q, target_h * target_w])  # [B, Q, S]

        attn_bias = tf.fill(tf.shape(mask_bool_flat), -1e9)
        attn_bias = tf.where(mask_bool_flat, tf.zeros_like(attn_bias), attn_bias)

        # Handle empty masks to prevent NaN
        row_max = tf.reduce_max(attn_bias, axis=-1, keepdims=True)
        is_empty_row = tf.equal(row_max, -1e9)
        is_empty_row_broad = tf.broadcast_to(is_empty_row, tf.shape(attn_bias))
        attn_bias = tf.where(is_empty_row_broad, tf.zeros_like(attn_bias), attn_bias)

        return attn_bias

    def _prepare_mask2former_bias(self, mask_preds, img_shape=None, threshold=0.5):
        """
        Converts mask predictions into an additive attention mask bias for Mask2Former.

        This method resizes mask predictions to the target image shape using bilinear interpolation,
        creates a binary mask based on the threshold, and converts it into an additive bias
        (0 for valid regions, large negative value for invalid regions) for attention mechanisms.
        It also handles empty masks to prevent NaN values in softmax.
        """
        # mask_preds shape: (Batch, Num_Queries, H, W)

        # 1. Resize logic
        if img_shape is not None:
            # Transpose to (Batch, H, W, Num_Queries) for tf.image.resize
            mask_preds_t = tf.transpose(mask_preds, perm=[0, 2, 3, 1])

            # Use bilinear to align with standard implementations
            mask_preds_resized = tf.image.resize(
                mask_preds_t,
                size=img_shape,
                method='bilinear'
            )

            # Transpose back to (Batch, Num_Queries, H, W)
            mask_preds = tf.transpose(mask_preds_resized, perm=[0, 3, 1, 2])

        # 2. Flatten spatial dimensions -> (Batch, Num_Queries, H*W)
        batch_size = tf.shape(mask_preds)[0]
        num_queries = tf.shape(mask_preds)[1]

        # Reshape to (Batch, Num_Queries, H*W)
        mask_flat = tf.reshape(mask_preds, (batch_size, num_queries, -1))

        # Create binary mask and invert to bias
        binary_mask = tf.cast(mask_flat > threshold, tf.float32)
        large_neg = -1e9
        attn_bias = (1.0 - binary_mask) * large_neg

        # Handle empty masks to prevent NaN
        mask_is_not_empty = tf.reduce_any(binary_mask > 0.5, axis=-1, keepdims=True)
        attn_bias = tf.where(mask_is_not_empty, attn_bias, tf.zeros_like(attn_bias))

        # Expand for heads: [B, 1, Q, S]
        attn_bias = tf.expand_dims(attn_bias, axis=1)

        return attn_bias

    def call(self,
             tgt,
             memory_list,         # list of [B, S_l, C]
             memory_pos_list,     # list of [B, S_l, C]
             decoder_shapes,      # [L_dec, 2] : (H_l, W_l)
             query_pos=None,
             mask_features=None,  # [B, H/4, W/4, C]
             mask_embed_fn=None,
             training=False):
        """
        Args:
            tgt: [B, Q, C] initial query embeddings (query content)
            memory_list: list of encoder feature tensors at different scales (length = L)
                         Each memory tensor is [B, S_l, C] with S_l = H_l*W_l flattened.
            memory_pos_list: list of positional encodings for each memory tensor [B, S_l, C].
            decoder_shapes: Tensor of shape [L, 2] with spatial H,W for each feature level.
                            (These are used to resize masks to the correct resolution per level.)
            query_pos: [B, Q, C] query positional embeddings.
            mask_features: [B, Hm, Wm, C] low-resolution mask feature map (e.g., stride-4 features for mask prediction).
            mask_embed_fn: function or layer to produce mask embeddings from decoder output (e.g., `self.mask_embed`).
            training: bool.
        Returns:
            intermediate_states: list of decoder outputs [B, Q, C] from each decoder layer (including initial if applicable).
        """
        B = tf.shape(tgt)[0]
        Q = tf.shape(tgt)[1]
        #num_levels = tf.shape(decoder_shapes)[0]
        num_levels = len(memory_list)

        intermediate_states = []
        # Prepare initial cross-attention mask
        if mask_features is not None and mask_embed_fn is not None:
            initial_mask_embed = mask_embed_fn(tgt)  # [B, Q, C]
            initial_mask_logits = tf.einsum('bqc,bhwc->bqhw', initial_mask_embed,
                                            mask_features)
            target_h = decoder_shapes[0, 0]
            target_w = decoder_shapes[0, 1]
            cross_attn_bias = self._prepare_mask2former_bias(initial_mask_logits, img_shape=(target_h, target_w))
        else:
            cross_attn_bias = None

        # Decoder layers with masked cross-attention
        for i, layer in enumerate(self.layers):
            lvl = i % num_levels
            tgt = layer(
                tgt, memory_list[lvl],
                query_pos=query_pos, memory_pos=memory_pos_list[lvl],
                cross_attention_mask=cross_attn_bias, training=training
            )
            intermediate_states.append(tgt)

            # Prepare mask for next layer
            if mask_features is not None and mask_embed_fn is not None and i < self.num_layers - 1:
                mask_embed = mask_embed_fn(tgt)
                mask_logits_highres = tf.einsum('bqc,bhwc->bqhw', mask_embed, mask_features)
                next_lvl = (i + 1) % num_levels
                next_h = decoder_shapes[next_lvl, 0]
                next_w = decoder_shapes[next_lvl, 1]
                cross_attn_bias = self._prepare_mask2former_bias(mask_logits_highres, img_shape=(next_h, next_w))
            else:
                if i == self.num_layers - 1:
                    pass
                else:
                    cross_attn_bias = None

        return intermediate_states
