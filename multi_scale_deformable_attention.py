"""
Author: Pavel Timonin
Created: 2025-09-28
Description: This script contains classes and functions for Multi-Scale Deformable Attention.
"""

import tensorflow as tf
import math


def bilinear_sample_nhwc(images, x_norm, y_norm):
    """
    Performs bilinear sampling on NHWC images (TensorFlow native layout).

    Optimized to remove redundant transposes.

    Args:
        images (tf.Tensor): Input images of shape [B, H, W, C] (Channels Last).
        x_norm (tf.Tensor): Normalized x coordinates of shape [B, Lq, P].
        y_norm (tf.Tensor): Normalized y coordinates of shape [B, Lq, P].

    Returns:
        tf.Tensor: Sampled values of shape [B, C, Lq, P].
    """
    images = tf.convert_to_tensor(images)
    x_norm = tf.convert_to_tensor(x_norm, dtype=tf.float32)
    y_norm = tf.convert_to_tensor(y_norm, dtype=tf.float32)

    shape = tf.shape(images)
    B = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]

    Lq = tf.shape(x_norm)[1]
    P = tf.shape(x_norm)[2]

    # 1. Normalize -> pixel coordinates (align_corners=False)
    #    Maps [0, 1] to [-0.5, W-0.5]
    x = x_norm * tf.cast(W, tf.float32) - 0.5
    y = y_norm * tf.cast(H, tf.float32) - 0.5

    # Neighbors
    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    # Interpolation weights [B, Lq, P, 1] for broadcasting against Channels
    wa = tf.expand_dims((x1 - x) * (y1 - y), axis=-1)
    wb = tf.expand_dims((x1 - x) * (y - y0), axis=-1)
    wc = tf.expand_dims((x - x0) * (y1 - y), axis=-1)
    wd = tf.expand_dims((x - x0) * (y - y0), axis=-1)

    # 2. Validity masks (padding_mode='zeros')
    H_f = tf.cast(H, tf.float32)
    W_f = tf.cast(W, tf.float32)

    x0_valid = (x0 >= 0.0) & (x0 <= W_f - 1.0)
    x1_valid = (x1 >= 0.0) & (x1 <= W_f - 1.0)
    y0_valid = (y0 >= 0.0) & (y0 <= H_f - 1.0)
    y1_valid = (y1 >= 0.0) & (y1 <= H_f - 1.0)

    # Convert to safe indices
    x0_idx = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
    x1_idx = tf.clip_by_value(tf.cast(x1, tf.int32), 0, W - 1)
    y0_idx = tf.clip_by_value(tf.cast(y0, tf.int32), 0, H - 1)
    y1_idx = tf.clip_by_value(tf.cast(y1, tf.int32), 0, H - 1)

    # Pre-calculate batch indices once
    batch_idx = tf.reshape(tf.range(B, dtype=tf.int32), [B, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, Lq, P])  # [B, Lq, P]

    def gather_nhwc(x_idx, y_idx, valid_x, valid_y):
        # Stack indices: [B, Lq, P, 3] -> (batch, y, x)
        indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
        # gather_nd on NHWC is efficient
        vals = tf.gather_nd(images, indices)  # [B, Lq, P, C]

        # Apply mask
        valid = tf.cast(valid_x & valid_y, vals.dtype)
        return vals * tf.expand_dims(valid, axis=-1)

    Ia = gather_nhwc(x0_idx, y0_idx, x0_valid, y0_valid)
    Ib = gather_nhwc(x0_idx, y1_idx, x0_valid, y1_valid)
    Ic = gather_nhwc(x1_idx, y0_idx, x1_valid, y0_valid)
    Id = gather_nhwc(x1_idx, y1_idx, x1_valid, y1_valid)

    # Weighted Sum
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # [B, Lq, P, C]

    # 3. Center validity (if original point was outside [0,1])
    x_center_valid = (x_norm >= 0.0) & (x_norm <= 1.0)
    y_center_valid = (y_norm >= 0.0) & (y_norm <= 1.0)
    center_valid = tf.cast(x_center_valid & y_center_valid, out.dtype)
    out = out * tf.expand_dims(center_valid, axis=-1)

    # Convert back to [B, C, Lq, P] to match MSDeformAttn expected interface
    out = tf.transpose(out, [0, 3, 1, 2])
    return out


def ms_deform_attn_core_tf(value, value_spatial_shapes, sampling_locations,
                           attention_weights, n_levels):
    """
    Core multi-scale deformable attention computation.

    Optimized to keep 'value' in NHWC format to avoid transposes.

    Args:
        value (tf.Tensor): Value features of shape [N, S, M, D_head].
        value_spatial_shapes (tf.Tensor): Spatial shapes of each level, shape [L, 2].
        sampling_locations (tf.Tensor): Sampling locations of shape [N, Lq, M, L, P, 2].
        attention_weights (tf.Tensor): Attention weights of shape [N, Lq, M, L, P].
        n_levels (int): Number of feature levels.

    Returns:
        tf.Tensor: Output features of shape [N, Lq, M*D_head].
    """
    value = tf.convert_to_tensor(value)
    value_spatial_shapes = tf.cast(value_spatial_shapes, tf.int32)
    sampling_locations = tf.convert_to_tensor(sampling_locations, dtype=tf.float32)
    attention_weights = tf.convert_to_tensor(attention_weights, dtype=value.dtype)

    N = tf.shape(value)[0]
    S = tf.shape(value)[1]
    M = tf.shape(value)[2]
    D_head = tf.shape(value)[3]

    Lq = tf.shape(sampling_locations)[1]
    P = tf.shape(sampling_locations)[4]

    # Split value into per-level tensors along S dimension
    hw = value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]
    value_list = tf.split(value, hw, axis=1)

    sampling_value_list = []

    for lid in range(n_levels):
        H_l = value_spatial_shapes[lid, 0]
        W_l = value_spatial_shapes[lid, 1]

        # OPTIMIZATION: Prepare in NHWC format [N*M, H, W, D_head]
        # value: [N, H*W, M, D]
        v_l = value_list[lid]
        # Transpose to [N, M, H*W, D] so we can merge N*M
        v_l = tf.transpose(v_l, [0, 2, 1, 3])
        v_l = tf.reshape(v_l, [N * M, H_l, W_l, D_head])  # NHWC

        # Sampling Grid
        grid_l = sampling_locations[:, :, :, lid, :, :]  # [N, Lq, M, P, 2]
        grid_l = tf.transpose(grid_l, [0, 2, 1, 3, 4])  # [N, M, Lq, P, 2]
        grid_l = tf.reshape(grid_l, [N * M, Lq, P, 2])

        x_norm = grid_l[..., 0]
        y_norm = grid_l[..., 1]

        # Call NHWC sampler
        sampling_value_l = bilinear_sample_nhwc(v_l, x_norm, y_norm)  # Returns [N*M, D, Lq, P]
        sampling_value_list.append(sampling_value_l)

    # Stack levels: [N*M, D_head, Lq, L, P]
    sampling_values = tf.stack(sampling_value_list, axis=-2)

    # Flatten L and P: [N*M, D_head, Lq, L*P]
    sampling_values = tf.reshape(sampling_values, [N * M, D_head, Lq, -1])

    # Attention weights: [N, Lq, M, L, P] -> [N*M, 1, Lq, L*P]
    attn = tf.transpose(attention_weights, [0, 2, 1, 3, 4])
    attn = tf.reshape(attn, [N * M, 1, Lq, -1])

    # Weighted sum
    out = tf.reduce_sum(sampling_values * attn, axis=-1)  # [N*M, D, Lq]

    # [N*M, D, Lq] -> [N, M, D, Lq] -> [N, Lq, M, D] -> [N, Lq, M*D]
    out = tf.reshape(out, [N, M * D_head, Lq])
    out = tf.transpose(out, [0, 2, 1])

    return out


class MSDeformAttn(tf.keras.layers.Layer):
    """
    Multi-Scale Deformable Attention Layer.

    Applies the deformable attention mechanism across multiple feature levels,
    enabling the model to attend to sparse spatial locations determined by
    learnable offsets.

    Args:
        d_model (int): Model dimension. Defaults to 256.
        n_levels (int): Number of feature levels. Defaults to 4.
        n_heads (int): Number of attention heads. Defaults to 8.
        n_points (int): Number of sampling points per head per level. Defaults to 4.
        reference_points_dim (int): Dimension of reference points (2 or 4). Defaults to 2.
        **kwargs: Additional keyword arguments for the base Layer class.

    Raises:
        ValueError: If d_model is not divisible by n_heads or reference_points_dim is not 2 or 4.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4,
                 reference_points_dim=2, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")
        if reference_points_dim not in (2, 4):
            raise ValueError(f"reference_points_dim must be 2 or 4, got {reference_points_dim}")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_per_head = d_model // n_heads
        self.reference_points_dim = reference_points_dim

    def _sampling_offsets_bias_init(self, shape, dtype=None):
        """
        Initializes sampling offsets bias with a grid pattern.

        Args:
            shape (tuple): Shape of the bias tensor.
            dtype (tf.DType, optional): Data type. Defaults to tf.float32.

        Returns:
            tf.Tensor: Initialized bias tensor.
        """
        if dtype is None: dtype = tf.float32
        thetas = tf.cast(tf.range(self.n_heads), dtype) * (2.0 * math.pi / float(self.n_heads))
        grid = tf.stack([tf.cos(thetas), tf.sin(thetas)], axis=-1)
        grid = grid / tf.reduce_max(tf.abs(grid), axis=-1, keepdims=True)
        grid = tf.reshape(grid, [self.n_heads, 1, 1, 2])
        points = tf.cast(tf.range(1, self.n_points + 1), dtype)
        points = tf.reshape(points, [1, 1, -1, 1])
        grid = grid * points
        grid = tf.tile(grid, [1, self.n_levels, 1, 1])
        return tf.cast(tf.reshape(grid, [-1]), dtype)

    def build(self, input_shape):
        """
        Builds the layer by creating trainable weights.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        self.sampling_offsets_kernel = self.add_weight(
            "sampling_offsets_kernel",
            shape=(self.d_model, self.n_heads * self.n_levels * self.n_points * 2),
            initializer=tf.zeros_initializer(), trainable=True)
        self.sampling_offsets_bias = self.add_weight(
            "sampling_offsets_bias",
            shape=(self.n_heads * self.n_levels * self.n_points * 2,),
            initializer=self._sampling_offsets_bias_init, trainable=True)

        self.attention_weights_kernel = self.add_weight(
            "attention_weights_kernel",
            shape=(self.d_model, self.n_heads * self.n_levels * self.n_points),
            initializer=tf.zeros_initializer(), trainable=True)
        self.attention_weights_bias = self.add_weight(
            "attention_weights_bias",
            shape=(self.n_heads * self.n_levels * self.n_points,),
            initializer=tf.zeros_initializer(), trainable=True)

        self.value_proj_kernel = self.add_weight(
            "value_proj_kernel", shape=(self.d_model, self.d_model),
            initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.value_proj_bias = self.add_weight(
            "value_proj_bias", shape=(self.d_model,),
            initializer=tf.zeros_initializer(), trainable=True)

        self.output_proj_kernel = self.add_weight(
            "output_proj_kernel", shape=(self.d_model, self.d_model),
            initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.output_proj_bias = self.add_weight(
            "output_proj_bias", shape=(self.d_model,),
            initializer=tf.zeros_initializer(), trainable=True)
        super().build(input_shape)

    def call(self, query, reference_points, input_flatten, input_spatial_shapes,
             input_level_start_index=None, input_padding_mask=None):
        """
        Applies multi-scale deformable attention.

        Args:
            query (tf.Tensor): Query features of shape [N, Len_q, d_model].
            reference_points (tf.Tensor): Reference points of shape [N, Len_q, n_levels, 2] or [N, Len_q, n_levels, 4].
            input_flatten (tf.Tensor): Flattened input features of shape [N, Len_in, d_model].
            input_spatial_shapes (tf.Tensor): Spatial shapes of each level, shape [n_levels, 2].
            input_level_start_index (tf.Tensor, optional): Starting index for each level. Defaults to None.
            input_padding_mask (tf.Tensor, optional): Padding mask. Defaults to None.

        Returns:
            tf.Tensor: Output features of shape [N, Len_q, d_model].
        """
        query = tf.cast(query, tf.float32)
        reference_points = tf.cast(reference_points, tf.float32)
        input_flatten = tf.cast(input_flatten, tf.float32)
        input_spatial_shapes = tf.cast(input_spatial_shapes, tf.int32)

        N = tf.shape(query)[0]
        Len_q = tf.shape(query)[1]
        Len_in = tf.shape(input_flatten)[1]

        value = tf.matmul(input_flatten, self.value_proj_kernel) + self.value_proj_bias
        if input_padding_mask is not None:
            pad_mask = tf.cast(input_padding_mask, value.dtype)
            value = value * tf.expand_dims(1.0 - pad_mask, axis=-1)
        value = tf.reshape(value, [N, Len_in, self.n_heads, self.d_per_head])

        sampling_offsets = tf.matmul(query, self.sampling_offsets_kernel) + self.sampling_offsets_bias
        sampling_offsets = tf.reshape(sampling_offsets, [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2])

        attention_weights = tf.matmul(query, self.attention_weights_kernel) + self.attention_weights_bias
        attention_weights = tf.nn.softmax(
            tf.reshape(attention_weights, [N, Len_q, self.n_heads, self.n_levels * self.n_points]), axis=-1)
        attention_weights = tf.reshape(attention_weights, [N, Len_q, self.n_heads, self.n_levels, self.n_points])

        if self.reference_points_dim == 2:
            H_l = tf.cast(input_spatial_shapes[:, 0], tf.float32)
            W_l = tf.cast(input_spatial_shapes[:, 1], tf.float32)
            offset_norm = tf.stack([W_l, H_l], axis=-1)
            sampling_locations = (reference_points[:, :, tf.newaxis, :, tf.newaxis, :] +
                                  sampling_offsets / offset_norm[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, :])
        elif self.reference_points_dim == 4:
            sampling_locations = (reference_points[:, :, tf.newaxis, :, tf.newaxis, :2] +
                                  sampling_offsets / float(self.n_points) * reference_points[
                                      :, :, tf.newaxis, :, tf.newaxis, 2:] * 0.5)

        output = ms_deform_attn_core_tf(value, input_spatial_shapes, sampling_locations, attention_weights,
                                        self.n_levels)
        output = tf.matmul(output, self.output_proj_kernel) + self.output_proj_bias
        return output