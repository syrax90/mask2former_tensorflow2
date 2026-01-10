"""
Author: Pavel Timonin
Created: 2025-10-17
Description: This script contains classes and functions of COCO dataset optimized for tf.Dataset.
"""


import os
from typing import Optional, Tuple
from coco_dataset import compute_scale_ranges

import tensorflow as tf

# Feature spec that matches the COCO TFRecord format
_FEATURES = {
    # image-level fields
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string),
    "image/id": tf.io.FixedLenFeature([], tf.int64),
    "image/format": tf.io.FixedLenFeature([], tf.string),

    # per-object fields
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/area": tf.io.VarLenFeature(tf.float32),
    "image/object/category_id": tf.io.VarLenFeature(tf.int64),
    "image/object/iscrowd": tf.io.VarLenFeature(tf.int64),
    "image/object/mask_png": tf.io.VarLenFeature(tf.string),
}

def sparse_to_dense_1d(v, dtype):
    """
    Convert a VarLen sparse tensor to a 1D dense tensor (length N).

    Args:
      v: `tf.SparseTensor`. A rank-1 sparse tensor.
      dtype: `tf.DType`. The desired dtype of the output.

    Returns:
      Tensor: Dense 1D tensor with shape [N] and dtype `dtype`.
    """
    return tf.cast(tf.sparse.to_dense(v), dtype)

# -----------------------------
# Data augmentations (graph mode, TF-only)
# -----------------------------
def maybe_hflip(img, masks, bboxes):
    """
    Randomly applies a horizontal flip to image, masks, and boxes (p=0.5).

    Args:
      img: Tensor [H, W, C] (uint8). Image.
      masks: Tensor [N, H, W] (uint8). Per-instance binary masks aligned to `img`.
      bboxes: Tensor [N, 4] (float32). Boxes in (x, y, w, h) format in the same
        coordinate space as `img`.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - img_f: Tensor [H, W, C] (uint8). Possibly flipped image.
        - masks_f: Tensor [N, H, W] (uint8). Possibly flipped masks.
        - b_new: Tensor [N, 4] (float32). Updated boxes after flip.

    Notes:
      * Applies with probability 0.5.
      * Boxes are mirrored around the image center by updating x: `x' = W - x - w`.
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        # Flip image and masks
        img_f = tf.image.flip_left_right(img)
        masks_f = tf.reverse(masks, axis=[2])  # [N,H,W], flip width

        # Adjust boxes
        W = tf.cast(tf.shape(img)[1], tf.float32)
        x, y, bw, bh = tf.unstack(bboxes, axis=1)
        x_new = W - x - bw
        b_new = tf.stack([x_new, y, bw, bh], axis=1)
        return img_f, masks_f, b_new
    def no():
        return img, masks, bboxes
    return tf.cond(do, yes, no)

def maybe_brightness(img):
    """
    Randomly jitters brightness by a multiplicative factor in [-20%, +20%] (p=0.5).

    Args:
      img: Tensor [H, W, C] (uint8). Image in range [0, 255].

    Returns:
      Tensor: Image of shape [H, W, C] (uint8) with brightness possibly adjusted.

    Notes:
      * Applies with probability 0.5.
      * The factor is sampled uniformly from [0.8, 1.2] and values are clipped to [0, 255].
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        factor = 1.0 + (tf.random.uniform([], -0.2, 0.2))
        img_f32 = tf.cast(img, tf.float32) * factor
        img_f32 = tf.clip_by_value(img_f32, 0.0, 255.0)
        return tf.cast(img_f32, tf.uint8)
    def no():
        return img
    return tf.cond(do, yes, no)

def maybe_scale(img, masks, bboxes):
    """
    Randomly scales image, masks, and boxes uniformly (p=0.5).

    The scale factor `s` is sampled from [0.8, 1.2]. Images are resized with bilinear
    interpolation; masks use nearest neighbor. Boxes are scaled by `s`.

    Args:
      img: Tensor [H, W, C] (uint8). Input image.
      masks: Tensor [N, H, W] (uint8). Per-instance masks aligned with `img`.
      bboxes: Tensor [N, 4] (float32). Boxes (x, y, w, h) in `img` coordinates.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - img_rs: Tensor [⌊H*s⌉, ⌊W*s⌉, C] (uint8). Resized image.
        - masks_rs: Tensor [N, ⌊H*s⌉, ⌊W*s⌉] (uint8). Resized masks.
        - b_new: Tensor [N, 4] (float32). Scaled boxes.

    Notes:
      * Applies with probability 0.5.
      * Image values are clipped to [0, 255] after bilinear resize and rounding.
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        s = tf.random.uniform([], 0.8, 1.2)

        # New size
        orig_hw = tf.cast(tf.shape(img)[:2], tf.float32)  # [H, W]
        new_hw = tf.cast(tf.round(orig_hw * s), tf.int32)
        new_h = new_hw[0]
        new_w = new_hw[1]

        # Resize image (bilinear) and masks (nearest)
        img_f32 = tf.cast(img, tf.float32)
        img_rs  = tf.image.resize(img_f32, size=[new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
        img_rs  = tf.clip_by_value(img_rs, 0.0, 255.0)
        img_rs  = tf.cast(tf.round(img_rs), tf.uint8)

        # Resize all masks at once
        masks_ch = tf.expand_dims(masks, axis=-1)                         # [N,H,W,1]
        masks_rs = tf.image.resize(masks_ch, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        masks_rs = tf.squeeze(masks_rs, axis=-1)                             # [N,H,W]
        masks_rs = tf.cast(masks_rs, tf.uint8)

        # Scale boxes and areas
        b_new = bboxes * tf.stack([s, s, s, s])  # (x,y,w,h) * s

        return img_rs, masks_rs, b_new
    def no():
        return img, masks, bboxes
    return tf.cond(do, yes, no)

def maybe_random_crop(img, masks, bboxes, cat_ids):
    """
    Applies a random crop up to 20% per side, updating masks/boxes/categories (p=0.5).

    A rectangular crop is sampled with left/top margins in [0, 0.2*W/H] and
    right/bottom margins in [0, 0.2*W/H]. Boxes are translated to the crop
    coordinate frame, clipped, and instances with very small resulting boxes
    (<=1 px width or height) are removed. The function keeps `cat_ids` and `masks`
    aligned with the filtered boxes.

    Args:
      img: Tensor [H, W, C] (uint8). Input image.
      masks: Tensor [N, H, W] (uint8). Per-instance masks.
      bboxes: Tensor [N, 4] (float32). Boxes (x, y, w, h) in image coords.
      cat_ids: Tensor [N] (int32). Category id per instance.

    Returns:
      Tuple[Tensor, Tensor, Tensor, Tensor]:
        - img_cr: Tensor [Hc, Wc, C] (uint8). Cropped image.
        - m_new: Tensor [N', Hc, Wc] (uint8). Cropped masks for kept instances.
        - b_new: Tensor [N', 4] (float32). Boxes translated/clipped to the crop.
        - c_new: Tensor [N'] (int32). Category ids for kept instances.

    Notes:
      * Applies with probability 0.5.
      * Keeps only boxes with width > 1 and height > 1 in pixels after cropping.
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        H = tf.shape(img)[0]
        W = tf.shape(img)[1]
        Hf = tf.cast(H, tf.float32)
        Wf = tf.cast(W, tf.float32)

        max_crop_x = tf.cast(tf.floor(Wf * 0.2), tf.int32)
        max_crop_y = tf.cast(tf.floor(Hf * 0.2), tf.int32)

        # Sample crop bounds: ensure left <= right and top <= bottom
        x1 = tf.random.uniform([], minval=0,              maxval=max_crop_x + 1, dtype=tf.int32)
        y1 = tf.random.uniform([], minval=0,              maxval=max_crop_y + 1, dtype=tf.int32)
        x2 = tf.random.uniform([], minval=W - max_crop_x, maxval=W + 1,         dtype=tf.int32)
        y2 = tf.random.uniform([], minval=H - max_crop_y, maxval=H + 1,         dtype=tf.int32)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Crop image and masks
        img_cr = tf.slice(img,   [y1, x1, 0], [crop_h, crop_w, -1])
        masks_cr = tf.slice(masks, [0, y1, x1], [-1, crop_h, crop_w])  # [N, crop_h, crop_w]

        # Adjust boxes to crop region, clip, and filter
        x, y, bw, bh = tf.unstack(bboxes, axis=1)
        x1f = tf.cast(x1, tf.float32)
        y1f = tf.cast(y1, tf.float32)
        cwf = tf.cast(crop_w, tf.float32)
        chf = tf.cast(crop_h, tf.float32)

        nx = tf.maximum(0.0, x - x1f)
        ny = tf.maximum(0.0, y - y1f)
        nw = tf.maximum(0.0, tf.minimum(bw, cwf - nx))
        nh = tf.maximum(0.0, tf.minimum(bh, chf - ny))

        keep = tf.logical_and(nw > 1.0, nh > 1.0)  # discard tiny/invalid boxes

        # Apply mask to per-instance tensors
        b_new = tf.boolean_mask(tf.stack([nx, ny, nw, nh], axis=1), keep)
        c_new = tf.boolean_mask(cat_ids,   keep)
        m_new = tf.boolean_mask(masks_cr,  keep, axis=0)


        return img_cr, m_new, b_new, c_new
    def no():
        return img, masks, bboxes, cat_ids
    return tf.cond(do, yes, no)

@tf.function
def parse_example(
        serialized,
        target_height,
        target_width,
        scale,
        augment):
    """
    Parses one TFRecord example and builds multi-scale SOLO training targets.

    This function:
      * Parses a single serialized example using `_FEATURES`.
      * Decodes the image (to RGB if needed) and per-instance masks (PNG).
      * Optionally applies augmentations (flip, brightness, random crop).
      * Resizes image to `(target_height, target_width)` and masks via nearest.
      * Scales boxes to the resized image coordinate frame.
      * For each SOLO grid size and its corresponding `scale_range`, generates
        per-scale targets via `generate_solo_targets_single_scale`, then
        concatenates category targets (flattened per scale) and mask targets
        (concatenated along channel axis).

    Args:
      serialized: Scalar string Tensor. A single serialized `tf.train.Example`.
      target_height: Scalar int. Output image height.
      target_width: Scalar int. Output image width.
      scale: Scalar float. Feature-map downscale (e.g., 0.25 -> 1/4).
        sqrt(area) gating.
      augment: Scalar bool. If True, apply data augmentations.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - image_resized: Tensor [target_height, target_width, 3] (float32) in [0, 1].
        - cate_targets: Tensor [sum(S_i^2)] (int32). Concatenated category
          targets from all scales, flattened per scale then concatenated.
        - mask_targets: Tensor [Hf, Wf, sum(S_i^2)] (uint8). Concatenated
          per-cell masks across all scales. Hf/Wf match the feature size for the
          provided `scale`.

    Notes:
      * If there are zero instances, shapes are still well-defined: category
        targets will be a concatenation of -1-filled grids; mask targets will be
        all zeros.
    """

    ex = tf.io.parse_single_example(serialized, _FEATURES)

    # Scalars
    img_enc = ex["image/encoded"]                  # bytes

    # Variable-length (per-object) -> dense 1D tensors
    xmin = sparse_to_dense_1d(ex["image/object/bbox/xmin"], tf.float32)
    ymin = sparse_to_dense_1d(ex["image/object/bbox/ymin"], tf.float32)
    xmax = sparse_to_dense_1d(ex["image/object/bbox/xmax"], tf.float32)
    ymax = sparse_to_dense_1d(ex["image/object/bbox/ymax"], tf.float32)

    x = xmin
    y = ymin
    w = (xmax - xmin)
    h = (ymax - ymin)

    cat_ids   = sparse_to_dense_1d(ex["image/object/category_id"], tf.int32)
    mask_pngs = sparse_to_dense_1d(ex["image/object/mask_png"], tf.string)

    # Stack boxes as [N, 4] in (x, y, w, h) format
    bboxes = tf.stack([x, y, w, h], axis=1) if tf.size(xmin) > 0 else tf.zeros([0, 4], tf.float32)

    # Decode image to ensure 3 channels
    img = tf.io.decode_image(img_enc, expand_animations=False)  # uint8, shape [H,W,C]
    # Ensure we have 3 channels (COCO images should be RGB)
    img = tf.cond(tf.shape(img)[-1] == 3,
                  lambda: img,
                  lambda: tf.image.grayscale_to_rgb(img))

    # Decode each per-object PNG into [H, W] uint8;
    def _decode_one(png_bytes):
        m = tf.io.decode_png(png_bytes, channels=1)  # [H,W,1]
        return tf.squeeze(m, axis=-1)  # [H,W]

    masks = tf.map_fn(_decode_one, mask_pngs, fn_output_signature=tf.uint8) # shape [N, H, W]

    def _apply_augmentation():
        # Horizontal flip
        img_aug, masks_aug, bboxes_aug = maybe_hflip(img, masks, bboxes)

        # Brightness jitter (+/-20%)
        img_aug = maybe_brightness(img_aug)

        # Random scaling (0.8x–1.2x)
        #img_aug, masks_aug, bboxes_aug = maybe_scale(img_aug, masks_aug, bboxes_aug)

        # Random crop (≤20% each side); updates and filters instance-aligned tensors
        img_aug, masks_aug, bboxes_aug, cat_ids_aug = maybe_random_crop(
            img_aug, masks_aug, bboxes_aug, cat_ids
        )

        return img_aug, masks_aug, bboxes_aug, cat_ids_aug

    img, masks, bboxes, cat_ids = tf.cond(augment, _apply_augmentation, lambda: (img, masks, bboxes, cat_ids))

    # Resize (bilinear by default; set method if you need a match)
    image_resized = tf.image.resize(img, size=(target_height, target_width), method="bilinear", antialias=True)

    # Convert to float32 in [0, 1]
    image_resized = tf.cast(image_resized, tf.float32) / 255.0

    # resize masks to (target_height, target_width) using nearest neighbor ===
    # Vectorized resize over batch dimension; works even if N == 0.
    target_mask_height = tf.cast(tf.round(target_height * scale), tf.int32)
    target_mask_width = tf.cast(tf.round(target_width * scale), tf.int32)
    masks_resized = tf.image.resize(
        tf.expand_dims(tf.cast(masks, tf.float32), axis=-1),  # [N,H,W,1]
        size=(target_mask_height, target_mask_width),
        method="nearest"
    )

    masks_resized = masks_resized / 255.0

    masks_resized = tf.cast(tf.round(tf.squeeze(masks_resized, axis=-1)), tf.uint8)  # [N,new_h,new_w]
    masks_resized = tf.transpose(masks_resized, perm=[1, 2, 0])  # [new_h,new_w,N] for later use
    cat_ids = cat_ids - 1  # convert to 0-based category ids

    return image_resized, cat_ids, masks_resized

def create_coco_tfrecord_dataset(
    train_tfrecord_directory: str,
    target_size: Tuple[int, int],
    batch_size: int,
    scale: float = 2.5,
    deterministic: bool = False,
    augment: bool = True,
    shuffle_buffer_size: Optional[int] = None,
    number_images: Optional[int] = None
) -> tf.data.Dataset:
    """Creates a `tf.data.Dataset` from COCO TFRecord shards and emits SOLO targets.

    This utility:
      * Scans a directory for `*.tfrecord` shards.
      * Builds a streaming `TFRecordDataset`.
      * Optionally shuffles and/or limits the number of examples.
      * Parses each example and constructs multi-scale SOLO targets via `parse_example`.
      * Batches and prefetches the dataset.

    Args:
      train_tfrecord_directory: Path to directory containing TFRecord shards.
      target_size: Tuple[int, int]. Target (height, width) for image & mask resizing.
      batch_size: Batch size for the resulting dataset.
      scale: Float. Feature-map scale factor used in target generation (e.g., 2.5).
        Note: This is later passed to `parse_example` which expects a downscale
        factor (e.g., 0.25); ensure consistency with your pipeline.
      deterministic: If False (default), allow non-deterministic map parallelism.
      augment: If True (default), apply data augmentations in `parse_example`.
      shuffle_buffer_size: Optional shuffle buffer size. If provided, shuffling is enabled.
      number_images: Optional cap on the number of images to take from the stream.

    Returns:
      tf.data.Dataset: A dataset of batched tuples:
        - image_resized: [B, Ht, Wt, 3] (float32) in [0, 1]
        - cate_targets: [B, sum(S_i^2)] (int32)
        - mask_targets: [B, Hf, Wf, sum(S_i^2)] (uint8)
    """
    target_height, target_width = target_size
    augment_tf = tf.constant(augment)

    # Gather all shard paths (common suffixes)
    pattern = "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(train_tfrecord_directory, pattern))

    if not files:
        raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecord_directory}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))

    # Shuffle
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    if number_images is not None:
        ds = ds.take(number_images)

    # Parse
    ds = ds.map(lambda x: parse_example(x, target_height=target_height, target_width=target_width, scale=scale, augment=augment_tf),
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)

    ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [target_size[0], target_size[1], 3],  # image shape
            [None, ],  # cate_target shape (num_instances,)
            [int(round(target_size[0] * scale)), int(round(target_size[1] * scale)), None]
        # mask_target shape (feat_h, feat_w, num_instances)
        ),
        padding_values=(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(-1, dtype=tf.int32),
            tf.constant(0, dtype=tf.uint8),
        ),
        drop_remainder=True
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
