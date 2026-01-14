"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains classes and functions for Mask2Former model building.
"""

import tensorflow as tf
from multi_scale_deformable_attention import MSDeformAttn

class SinePositionEmbedding(tf.keras.layers.Layer):
    """
    Standard DETR 2D sine/cosine positional encoding, matching Mask2Former:
    normalize=True, scale=2*pi.

    Attributes:
        num_pos_feats (int): Number of positional features.
        temperature (float): Temperature parameter for sine/cosine modulation.
        scale (float): Scale factor for normalization.
    """

    def __init__(self, num_pos_feats=128, temperature=10000, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.temperature = float(temperature)
        if scale is None:
            scale = 2.0 * tf.experimental.numpy.pi
        self.scale = float(scale)

    def call(self, x, mask=None):
        """
        Generates 2D sine/cosine positional embeddings for input features.

        Args:
            x (tf.Tensor): Input feature map of shape [B, H, W, C].
            mask (tf.Tensor, optional): Boolean mask of shape [B, H, W] where True indicates padding. Defaults to None.

        Returns:
            tf.Tensor: Positional embeddings of shape [B, H, W, 2*num_pos_feats].
        """
        # x: [B, H, W, C]
        x = tf.convert_to_tensor(x)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        if mask is None:
            # not_mask: [B, H, W] = True everywhere
            not_mask = tf.ones([B, H, W], dtype=tf.bool)
        else:
            # mask: True for padding; we want ~mask
            not_mask = tf.logical_not(mask)

        not_mask_f = tf.cast(not_mask, tf.float32)

        # Cumsum along height and width
        y_embed = tf.cumsum(not_mask_f, axis=1)
        x_embed = tf.cumsum(not_mask_f, axis=2)

        eps = 1e-6
        y_last = y_embed[:, -1:, :] + eps
        x_last = x_embed[:, :, -1:] + eps

        y_embed = y_embed / y_last * self.scale
        x_embed = x_embed / x_last * self.scale

        dim_t = tf.range(self.num_pos_feats, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / float(self.num_pos_feats))

        pos_x = x_embed[..., tf.newaxis] / dim_t  # [B, H, W, Cpos]
        pos_y = y_embed[..., tf.newaxis] / dim_t

        pos_x = tf.stack(
            [tf.sin(pos_x[..., 0::2]), tf.cos(pos_x[..., 1::2])],
            axis=-1,
        )  # [B,H,W,Cpos/2,2]
        pos_y = tf.stack(
            [tf.sin(pos_y[..., 0::2]), tf.cos(pos_y[..., 1::2])],
            axis=-1,
        )

        pos_x = tf.reshape(pos_x, [B, H, W, self.num_pos_feats])
        pos_y = tf.reshape(pos_y, [B, H, W, self.num_pos_feats])

        pos = tf.concat([pos_y, pos_x], axis=-1)  # [B, H, W, 2*num_pos_feats]
        return pos


class DeformableTransformerEncoderLayer(tf.keras.layers.Layer):
    """
    One encoder layer using MSDeformAttn + FFN (like Deformable DETR / Mask2Former).

    Args:
        d_model (int): Model dimension. Defaults to 256.
        n_levels (int): Number of feature levels. Defaults to 4.
        n_heads (int): Number of attention heads. Defaults to 8.
        n_points (int): Number of sampling points. Defaults to 4.
        dim_feedforward (int): Dimension of FFN. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        dim_feedforward=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.self_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            name=self.name + "_ms_deform_attn",
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=self.name + "_norm1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=self.name + "_norm2")

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

        self.linear1 = tf.keras.layers.Dense(dim_feedforward, name=self.name + "_ffn1")
        self.linear2 = tf.keras.layers.Dense(d_model, name=self.name + "_ffn2")

        self.activation = tf.nn.relu

    @staticmethod
    def with_pos_embed(tensor, pos):
        """
        Adds positional embeddings to tensor if provided.

        Args:
            tensor (tf.Tensor): Input tensor.
            pos (tf.Tensor, optional): Positional embeddings to add.

        Returns:
            tf.Tensor: Tensor with positional embeddings added, or original tensor if pos is None.
        """
        return tensor if pos is None else tensor + pos

    def call(
        self,
        src,                    # [B, S, C]
        pos,                    # [B, S, C] positional embedding
        reference_points,       # [B, S, L, 2]
        spatial_shapes,         # [L, 2]
        level_start_index,      # [L]
        padding_mask=None,      # [B, S] or None
        training=False,
    ):
        """
        Applies multi-scale deformable attention and feed-forward network.

        Args:
            src (tf.Tensor): Source features of shape [B, S, C].
            pos (tf.Tensor): Positional embeddings of shape [B, S, C].
            reference_points (tf.Tensor): Reference points of shape [B, S, L, 2].
            spatial_shapes (tf.Tensor): Spatial shapes of each level, shape [L, 2].
            level_start_index (tf.Tensor): Starting index for each level, shape [L].
            padding_mask (tf.Tensor, optional): Padding mask of shape [B, S]. Defaults to None.
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Transformed features of shape [B, S, C].
        """

        # MSDeformAttn self-attention (multi-scale)
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=padding_mask,
        )

        src = src + self.dropout1(src2, training=training)
        src = self.norm1(src)

        # FFN (matching Deformable DETR ordering)
        src2 = self.linear2(
            self.dropout2(
                self.activation(self.linear1(src)),
                training=training,
            )
        )
        src = src + self.dropout3(src2, training=training)
        src = self.norm2(src)

        return src

class DeformableTransformerEncoder(tf.keras.layers.Layer):
    """
    Stack of DeformableTransformerEncoderLayer (multi-scale deformable encoder).

    Args:
        num_layers (int): Number of encoder layers. Defaults to 6.
        d_model (int): Model dimension. Defaults to 256.
        n_levels (int): Number of feature levels. Defaults to 4.
        n_heads (int): Number of attention heads. Defaults to 8.
        n_points (int): Number of sampling points. Defaults to 4.
        dim_feedforward (int): Dimension of FFN. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        num_layers=6,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        dim_feedforward=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.n_levels = n_levels

        self.layers = [
            DeformableTransformerEncoderLayer(
                d_model=d_model,
                n_levels=n_levels,
                n_heads=n_heads,
                n_points=n_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                name=f"{self.name}_layer_{i}",
            )
            for i in range(num_layers)
        ]

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        """
        Generates reference points for multi-scale deformable attention.

        Ensures output is (x, y) compatible with bilinear sampling.

        Args:
            spatial_shapes (tf.Tensor): Spatial dimensions [L, 2] as (H_l, W_l) for each level.
            valid_ratios (tf.Tensor): Valid ratios [B, L, 2] as (ratio_h, ratio_w) for each level.

        Returns:
            tf.Tensor: Reference points of shape [B, S, L, 2] containing (x, y) coordinates.
        """
        spatial_shapes = tf.cast(spatial_shapes, tf.int32)
        valid_ratios = tf.cast(valid_ratios, tf.float32)

        B = tf.shape(valid_ratios)[0]
        L = tf.shape(spatial_shapes)[0]

        # We'll concatenate across levels along the "S" dimension.
        # Trick: write tensors as [HW, B, 2] into TensorArray and use ta.concat()
        ta = tf.TensorArray(tf.float32, size=L, infer_shape=False, element_shape=[None, None, 2])

        def cond(lvl, ta_):
            return lvl < L

        def body(lvl, ta_):
            # Generate grid for this level
            H = spatial_shapes[lvl, 0]
            W = spatial_shapes[lvl, 1]
            H_f = tf.cast(H, tf.float32)
            W_f = tf.cast(W, tf.float32)

            ref_y = tf.linspace(0.5, H_f - 0.5, H)
            ref_x = tf.linspace(0.5, W_f - 0.5, W)
            yy, xx = tf.meshgrid(ref_y, ref_x, indexing="ij")
            yy = tf.reshape(yy, [-1])
            xx = tf.reshape(xx, [-1])

            # Handle valid ratios
            ratio_h = valid_ratios[:, lvl, 0]
            ratio_w = valid_ratios[:, lvl, 1]
            yy_norm = yy[tf.newaxis, :] / (ratio_h[:, tf.newaxis] * H_f)
            xx_norm = xx[tf.newaxis, :] / (ratio_w[:, tf.newaxis] * W_f)

            # Stack as (x, y) coordinates
            ref = tf.stack([xx_norm, yy_norm], axis=-1)  # [B, HW, 2]
            ref = tf.transpose(ref, [1, 0, 2])  # [HW, B, 2]

            ta_ = ta_.write(lvl, ref)
            return lvl + 1, ta_

        # Reconstruct reference points: [B, S, L, 2]
        reference_points = tf.transpose(ta.concat(), [1, 0, 2])
        valid_ratios_wh = valid_ratios[:, tf.newaxis, :, ::-1]
        reference_points = reference_points[:, :, tf.newaxis, :] * valid_ratios_wh

        return reference_points

    def call(
        self,
        src,               # [B, S, C]
        spatial_shapes,    # [L, 2]
        level_start_index, # [L]
        valid_ratios,      # [B, L, 2]
        pos=None,          # [B, S, C]
        padding_mask=None, # [B, S] or None
        training=False,
    ):
        """
        Applies multi-scale deformable transformer encoding.

        Args:
            src (tf.Tensor): Source features of shape [B, S, C].
            spatial_shapes (tf.Tensor): Spatial shapes of each level, shape [L, 2].
            level_start_index (tf.Tensor): Starting index for each level, shape [L].
            valid_ratios (tf.Tensor): Valid ratios for each level, shape [B, L, 2].
            pos (tf.Tensor, optional): Positional embeddings of shape [B, S, C]. Defaults to None.
            padding_mask (tf.Tensor, optional): Padding mask of shape [B, S]. Defaults to None.
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Encoded features of shape [B, S, C].
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)

        for layer in self.layers:
            output = layer(
                output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                padding_mask=padding_mask,
                training=training,
            )
        return output


class MSDeformablePixelDecoder(tf.keras.layers.Layer):
    """
    Mask2Former-style pixel decoder with Multi-Scale Deformable Attention encoder.

    This layer takes multi-scale feature maps from the backbone, processes them with a deformable
    transformer encoder, and produces a high-resolution mask feature map and multi-level memory
    features for the decoder.

    Args:
        d_model (int): Model dimension. Defaults to 256.
        num_feature_levels (int): Number of input feature levels. Defaults to 4.
        transformer_num_feature_levels (int): Number of levels used in transformer. Defaults to 3.
        num_encoder_layers (int): Number of encoder layers. Defaults to 6.
        n_heads (int): Number of attention heads. Defaults to 8.
        n_points (int): Number of sampling points. Defaults to 4.
        dim_feedforward (int): Dimension of FFN. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        d_model=256,
        num_feature_levels=4,
        transformer_num_feature_levels=3,
        num_encoder_layers=6,
        n_heads=8,
        n_points=4,
        dim_feedforward=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        self.transformer_num_feature_levels = transformer_num_feature_levels

        # Per-level learnable embedding (like Deformable DETR)
        self.level_embed = self.add_weight(
            name=f"{self.name}_level_embed",
            shape=(self.transformer_num_feature_levels, self.d_model),
            initializer=tf.random_normal_initializer(stddev=1.0),
            trainable=True,
        )

        # 1x1 conv to project each level to d_model
        self.input_proj = [
            tf.keras.layers.Conv2D(
                d_model,
                kernel_size=1,
                padding="same",
                name=f"{self.name}_input_proj_{i}",
            )
            for i in range(num_feature_levels)
        ]

        # Optional 3x3 conv for FPN-style smoothing per level
        self.output_convs = [
            tf.keras.layers.Conv2D(
                d_model,
                kernel_size=3,
                padding="same",
                activation=None,
                name=f"{self.name}_output_conv_{i}",
            )
            for i in range(num_feature_levels)
        ]

        # Multi-scale deformable encoder
        self.encoder = DeformableTransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_levels=transformer_num_feature_levels,
            n_heads=n_heads,
            n_points=n_points,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            name=f"{self.name}_encoder",
        )

        self.pos_embedding = SinePositionEmbedding(num_pos_feats=d_model // 2)

    def call(self, features, training=False):
        """
        Processes multi-scale features through pixel decoder.

        Args:
            features (list): List of feature tensors [x2, x3, x4, x5], each of shape [B, H_l, W_l, C_in].
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - memory_list (list): List of flattened encoder outputs [B, S_l, C] for each decoder level.
                - memory_pos_list (list): List of positional encodings [B, S_l, C] for each decoder level.
                - decoder_shapes (tf.Tensor): Spatial shapes [L_dec, 2] for decoder levels.
                - mask_features (tf.Tensor): High-resolution mask features [B, Hm, Wm, C] at stride 4.
        """
        assert len(features) == self.num_feature_levels, (
                "Expected %d feature levels, got %d"
                % (self.num_feature_levels, len(features))
        )

        proj_feats = []
        batch_size = tf.shape(features[0])[0]

        # Project all 4 levels (p2..p5)
        for i in range(self.num_feature_levels):
            x = features[i]
            x_proj = self.input_proj[i](x)
            proj_feats.append(x_proj)

        # Build deformable-encoder inputs from p3, p4, p5
        enc_proj_feats = proj_feats[1:1 + self.transformer_num_feature_levels]
        pos_feats = []
        spatial_shapes_list = []
        level_element_nums = []

        # Project to common dimension and build positional encodings
        for i in range(self.transformer_num_feature_levels):
            x_proj = enc_proj_feats[i]

            pos_l = self.pos_embedding(x_proj)
            lvl_embed = self.level_embed[i][tf.newaxis, tf.newaxis, tf.newaxis, :]
            pos_l = pos_l + lvl_embed
            pos_feats.append(pos_l)

            H = tf.shape(x_proj)[1]
            W = tf.shape(x_proj)[2]
            spatial_shapes_list.append(tf.stack([H, W], axis=0))
            level_element_nums.append(H * W)

        spatial_shapes = tf.stack(spatial_shapes_list, axis=0)  # [L, 2]
        level_element_nums = tf.stack(level_element_nums, axis=0)  # [L]

        # Flatten and concat features across levels
        src_flatten_list = []
        pos_flatten_list = []

        for i in range(self.transformer_num_feature_levels):
            x = enc_proj_feats[i]
            pos = pos_feats[i]
            H = tf.shape(x)[1]
            W = tf.shape(x)[2]
            src_flatten_list.append(tf.reshape(x, [batch_size, H * W, self.d_model]))
            pos_flatten_list.append(tf.reshape(pos, [batch_size, H * W, self.d_model]))

        src = tf.concat(src_flatten_list, axis=1)  # [B, S, C]
        pos = tf.concat(pos_flatten_list, axis=1)  # [B, S, C]

        # level_start_index: [L]
        level_start_index = tf.concat(
            [
                tf.zeros((1,), dtype=level_element_nums.dtype),
                tf.math.cumsum(level_element_nums[:-1]),
            ],
            axis=0,
        )

        # Valid ratios and padding mask
        batch_size = tf.shape(src)[0]
        valid_ratios = tf.ones(
            [batch_size, self.transformer_num_feature_levels, 2],
            dtype=tf.float32,
        )
        padding_mask = None

        # Multi-scale deformable encoder
        memory = self.encoder(
            src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=pos,
            padding_mask=padding_mask,
            training=training,
        )  # [B, S, C]

        # Split encoder output back to feature maps
        encoded_feats = []
        start = 0
        for i in range(self.transformer_num_feature_levels):
            num_elems = level_element_nums[i]
            end = start + num_elems
            feat_i = memory[:, start:end, :]
            H_i = spatial_shapes[i, 0]
            W_i = spatial_shapes[i, 1]
            feat_i = tf.reshape(feat_i, [batch_size, H_i, W_i, self.d_model])
            encoded_feats.append(feat_i)
            start = end

        encoded_feats = [proj_feats[0]] + encoded_feats  # [p2, p3, p4, p5]

        # Top-down FPN-style fusion
        x = encoded_feats[-1]
        x = self.output_convs[-1](x)

        for level in reversed(range(self.num_feature_levels - 1)):
            target_h = tf.shape(encoded_feats[level])[1]
            target_w = tf.shape(encoded_feats[level])[2]
            x_upsampled = tf.image.resize(x, size=[target_h, target_w], method="bilinear")
            x = encoded_feats[level] + x_upsampled
            x = self.output_convs[level](x)

        mask_features = x

        # Build multi-scale memories for decoder
        decoder_feats = encoded_feats[1:]
        memory_list = []
        memory_pos_list = []
        decoder_shapes = []
        B = tf.shape(mask_features)[0]

        for feat in decoder_feats:
            H_l = tf.shape(feat)[1]
            W_l = tf.shape(feat)[2]
            decoder_shapes.append(tf.stack([H_l, W_l], axis=0))
            mem = tf.reshape(feat, [B, H_l * W_l, self.d_model])
            memory_list.append(mem)
            pos_l = self.pos_embedding(feat)
            pos_l = tf.reshape(pos_l, [B, H_l * W_l, self.d_model])
            memory_pos_list.append(pos_l)

        decoder_shapes = tf.stack(decoder_shapes, axis=0)
        return memory_list, memory_pos_list, decoder_shapes, mask_features
