"""
Author: Pavel Timonin
Created: 2025-09-28
Description: This script contains classes and functions of COCO dataset optimized for tf.Dataset.
"""

import os

from typing import Optional, Tuple
from pycocotools.coco import COCO
from reassign_categories import reassign_category_ids
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


class COCOAnalysis:
    """
    Helper class to analyze COCO annotation file and extract class information.

    Args:
        annotation_path (str): Path to the COCO annotation JSON file.
    """
    def __init__(self, annotation_path: str):
        self.coco = COCO(annotation_path)
        # reassign_category_ids modifies self.coco in-place and rebuilds the index
        reassign_category_ids(self.coco)
        self.category_ids = sorted(self.coco.getCatIds())
        self.categories = self.coco.loadCats(self.category_ids)

    def get_num_classes(self) -> int:
        """
        Get the number of classes defined in the annotation file.

        Returns:
            int: The number of classes.
        """
        return len(self.category_ids)

    def get_class_names(self) -> list:
        """
        Get the list of class names.

        Returns:
            list: A list of class names.
        """
        return [cat["name"] for cat in self.categories]

    def get_category_ids(self) -> list:
        """
        Get the list of category IDs.

        Returns:
            list: A list of category IDs.
        """
        return self.category_ids


def sparse_to_dense_1d(v, dtype):
    """
    Convert a VarLen sparse tensor to a 1D dense tensor (length N).

    Args:
        v (tf.SparseTensor): A rank-1 sparse tensor.
        dtype (tf.DType): The desired dtype of the output.

    Returns:
        tf.Tensor: Dense 1D tensor with shape [N] and dtype `dtype`.
    """
    return tf.cast(tf.sparse.to_dense(v), dtype)


# Data augmentations

def maybe_hflip(img, masks, bboxes):
    """
    Apply a horizontal flip to image, masks, and boxes randomly (p=0.5).

    Boxes are mirrored around the image center by updating x: `x' = W - x - w`.

    Args:
        img (tf.Tensor): Image tensor [H, W, C] (uint8).
        masks (tf.Tensor): Per-instance binary masks aligned to `img`, [N, H, W] (uint8).
        bboxes (tf.Tensor): Boxes in (x, y, w, h) format in the same coordinate space as `img`, [N, 4] (float32).

    Returns:
        tuple: A tuple containing:
            - img_f (tf.Tensor): Possibly flipped image [H, W, C] (uint8).
            - masks_f (tf.Tensor): Possibly flipped masks [N, H, W] (uint8).
            - b_new (tf.Tensor): Updated boxes after flip [N, 4] (float32).
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
    Jitter brightness randomly by a multiplicative factor in [-20%, +20%] (p=0.5).

    The factor is sampled uniformly from [0.8, 1.2] and values are clipped to [0, 255].

    Args:
        img (tf.Tensor): Image in range [0, 255], [H, W, C] (uint8).

    Returns:
        tf.Tensor: Image of shape [H, W, C] (uint8) with brightness possibly adjusted.
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


def maybe_random_crop(img, masks, bboxes, cat_ids):
    """
    Apply a random crop up to 20% per side, updating masks/boxes/categories (p=0.5).

    A rectangular crop is sampled with left/top margins in [0, 0.2*W/H] and
    right/bottom margins in [0, 0.2*W/H]. Boxes are translated to the crop
    coordinate frame, clipped, and instances with very small resulting boxes
    (<=1 px width or height) are removed. The function keeps `cat_ids` and `masks`
    aligned with the filtered boxes.

    Args:
        img (tf.Tensor): Input image [H, W, C] (uint8).
        masks (tf.Tensor): Per-instance masks [N, H, W] (uint8).
        bboxes (tf.Tensor): Boxes (x, y, w, h) in image coords [N, 4] (float32).
        cat_ids (tf.Tensor): Category id per instance [N] (int32).

    Returns:
        tuple: A tuple containing:
            - img_cr (tf.Tensor): Cropped image [Hc, Wc, C] (uint8).
            - m_new (tf.Tensor): Cropped masks for kept instances [N', Hc, Wc] (uint8).
            - b_new (tf.Tensor): Boxes translated/clipped to the crop [N', 4] (float32).
            - c_new (tf.Tensor): Category ids for kept instances [N'] (int32).
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
        x1 = tf.random.uniform([], minval=0, maxval=max_crop_x + 1, dtype=tf.int32)
        y1 = tf.random.uniform([], minval=0, maxval=max_crop_y + 1, dtype=tf.int32)
        x2 = tf.random.uniform([], minval=W - max_crop_x, maxval=W + 1, dtype=tf.int32)
        y2 = tf.random.uniform([], minval=H - max_crop_y, maxval=H + 1, dtype=tf.int32)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Crop image and masks
        img_cr = tf.slice(img, [y1, x1, 0], [crop_h, crop_w, -1])
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
        c_new = tf.boolean_mask(cat_ids, keep)
        m_new = tf.boolean_mask(masks_cr, keep, axis=0)

        return img_cr, m_new, b_new, c_new

    def no():
        return img, masks, bboxes, cat_ids

    return tf.cond(do, yes, no)


@tf.function
def _parse_example_base(
        serialized,
        target_height,
        target_width,
        scale,
        augment):
    """
    Parse one TFRecord example and build multi-scale Mask2Former training targets.

    This function parses a single serialized example, decodes the image and per-instance masks,
    optionally applies augmentations (flip, random crop, brightness), resizes the image and masks,
    scales boxes, and generates per-scale targets.
    It then concatenates category targets (flattened per scale) and mask targets
    (concatenated along channel axis).

    Args:
        serialized (tf.Tensor): Scalar string Tensor. A single serialized `tf.train.Example`.
        target_height (int): Output image height.
        target_width (int): Output image width.
        scale (float): Feature-map downscale factor (e.g., 0.25 -> 1/4) for sqrt(area) gating.
        augment (bool): If True, apply data augmentations.

    Returns:
        tuple: A tuple containing:
            - image_resized (tf.Tensor): Resized image [target_height, target_width, 3] (float32) in [0, 1].
            - cate_targets (tf.Tensor): Concatenated category targets from all scales [sum(S_i^2)] (int32).
            - mask_targets (tf.Tensor): Concatenated per-cell masks across all scales [Hf, Wf, sum(S_i^2)] (uint8).
            - image_id (tf.Tensor): Image ID.
            - original_height (tf.Tensor): Original image height.
            - original_width (tf.Tensor): Original image width.
    """

    ex = tf.io.parse_single_example(serialized, _FEATURES)

    # Scalars
    img_enc = ex["image/encoded"]  # bytes
    image_id = ex["image/id"]
    original_height = ex["image/height"]
    original_width = ex["image/width"]

    # Variable-length (per-object) -> dense 1D tensors
    xmin = sparse_to_dense_1d(ex["image/object/bbox/xmin"], tf.float32)
    ymin = sparse_to_dense_1d(ex["image/object/bbox/ymin"], tf.float32)
    xmax = sparse_to_dense_1d(ex["image/object/bbox/xmax"], tf.float32)
    ymax = sparse_to_dense_1d(ex["image/object/bbox/ymax"], tf.float32)

    x = xmin
    y = ymin
    w = (xmax - xmin)
    h = (ymax - ymin)

    cat_ids = sparse_to_dense_1d(ex["image/object/category_id"], tf.int32)
    mask_pngs = sparse_to_dense_1d(ex["image/object/mask_png"], tf.string)

    # Stack boxes as [N, 4] in (x, y, w, h) format
    if tf.size(xmin) > 0:
        bboxes = tf.stack([x, y, w, h], axis=1)
    else:
        bboxes = tf.zeros([0, 4], tf.float32)

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

    masks = tf.map_fn(_decode_one, mask_pngs, fn_output_signature=tf.uint8)  # shape [N, H, W]

    def _apply_augmentation():
        # Horizontal flip
        img_aug, masks_aug, bboxes_aug = maybe_hflip(img, masks, bboxes)

        # Random crop (≤20% each side); updates and filters instance-aligned tensors
        img_aug, masks_aug, bboxes_aug, cat_ids_aug = maybe_random_crop(
            img_aug, masks_aug, bboxes_aug, cat_ids
        )

        # Brightness jitter (+/-20%) - moved after crop to reducing processing
        img_aug = maybe_brightness(img_aug)

        return img_aug, masks_aug, bboxes_aug, cat_ids_aug

    img, masks, bboxes, cat_ids = tf.cond(augment, _apply_augmentation, lambda: (img, masks, bboxes, cat_ids))

    # Resize (bilinear by default; set method if you need a match)
    image_resized = tf.image.resize(img, size=(target_height, target_width), method="bilinear", antialias=True)

    # Preprocess for ResNet50 (requires inputs in [0, 255] range but specific distribution)
    # The original model expects "caffe" mode: BGR, zero-centered
    image_resized = tf.cast(image_resized, tf.float32)
    image_resized = tf.keras.applications.resnet50.preprocess_input(image_resized)

    # Resize masks to (target_height, target_width) using nearest neighbor
    target_mask_height = tf.cast(tf.round(target_height * scale), tf.int32)
    target_mask_width = tf.cast(tf.round(target_width * scale), tf.int32)

    masks_expanded = tf.expand_dims(masks, axis=-1)

    # Use Nearest Neighbor for masks to preserve binary nature logic better than bilinear
    masks_resized = tf.image.resize(
        masks_expanded,
        size=(target_mask_height, target_mask_width),
        method="nearest"
    ) # [N, new_h, new_w, 1]

    masks_resized = tf.squeeze(masks_resized, axis=-1) # [N, new_h, new_w]
    masks_resized = tf.cast(masks_resized, tf.uint8)

    # Convert to binary 0/1 (original masks are usually 0 or 255 or 0/1)
    # Ensuring it's 0/1
    masks_resized = tf.cast(masks_resized > 127, tf.uint8)

    masks_resized = tf.transpose(masks_resized, perm=[1, 2, 0])
    cat_ids = cat_ids - 1  # convert to 0-based category ids

    return image_resized, cat_ids, masks_resized, image_id, original_height, original_width


@tf.function
def parse_example(
        serialized,
        target_height,
        target_width,
        scale,
        augment):
    """
    Parse one TFRecord example and build multi-scale Mask2Former training targets.

    This function parses a single serialized example, decodes the image and per-instance masks,
    optionally applies augmentations (flip, random crop, brightness), resizes the image and masks,
    scales boxes, and generates per-scale targets.
    It then concatenates category targets (flattened per scale) and mask targets
    (concatenated along channel axis).

    Args:
        serialized (tf.Tensor): Scalar string Tensor. A single serialized `tf.train.Example`.
        target_height (int): Output image height.
        target_width (int): Output image width.
        scale (float): Feature-map downscale factor (e.g., 0.25 -> 1/4) for sqrt(area) gating.
        augment (bool): If True, apply data augmentations.

    Returns:
        tuple: A tuple containing:
            - image_resized (tf.Tensor): Resized image [target_height, target_width, 3] (float32) in [0, 1].
            - cate_targets (tf.Tensor): Concatenated category targets from all scales [sum(S_i^2)] (int32).
            - mask_targets (tf.Tensor): Concatenated per-cell masks across all scales [Hf, Wf, sum(S_i^2)] (uint8).
    """
    r = _parse_example_base(serialized, target_height, target_width, scale, augment)
    # Return first 3 elements: image_resized, cat_ids, masks_resized
    return r[0], r[1], r[2]


@tf.function
def parse_eval_example(
        serialized,
        target_height,
        target_width,
        scale,
        augment):
    """
    Parse one TFRecord example for evaluation, including metadata.

    Args:
        serialized (tf.Tensor): Scalar string Tensor. A single serialized `tf.train.Example`.
        target_height (int): Output image height.
        target_width (int): Output image width.
        scale (float): Feature-map downscale factor.
        augment (bool): If True, apply data augmentations.

    Returns:
        tuple: A tuple containing:
            - image_resized (tf.Tensor): Resized image [target_height, target_width, 3] (float32) in [0, 1].
            - cate_targets (tf.Tensor): Concatenated category targets from all scales [sum(S_i^2)] (int32).
            - mask_targets (tf.Tensor): Concatenated per-cell masks across all scales [Hf, Wf, sum(S_i^2)] (uint8).
            - image_id (tf.Tensor): Image ID.
            - original_height (tf.Tensor): Original image height.
            - original_width (tf.Tensor): Original image width.
    """
    # Return all 6 elements
    return _parse_example_base(serialized, target_height, target_width, scale, augment)



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
    """
    Create a `tf.data.Dataset` from COCO TFRecord shards and emit Mask2Former targets.

    This utility scans a directory for `*.tfrecord` shards, builds a streaming `TFRecordDataset`,
    optionally shuffles and/or limits the number of examples, parses each example,
    constructs multi-scale Mask2Former targets, and batches/prefetches the dataset.

    Args:
        train_tfrecord_directory (str): Path to directory containing TFRecord shards.
        target_size (tuple): Target (height, width) for image & mask resizing.
        batch_size (int): Batch size for the resulting dataset.
        scale (float, optional): Feature-map scale factor used in target generation (e.g., 2.5).
            Defaults to 2.5.
        deterministic (bool, optional): If False, allow non-deterministic map parallelism.
            Defaults to False.
        augment (bool, optional): If True, apply data augmentations in `parse_example`.
            Defaults to True.
        shuffle_buffer_size (int, optional): Optional shuffle buffer size. If provided, shuffling is enabled.
            Defaults to None.
        number_images (int, optional): Optional cap on the number of images to take from the stream.
            Defaults to None.

    Returns:
        tf.data.Dataset: A dataset of batched tuples:
            - image_resized (tf.Tensor): [B, Ht, Wt, 3] (float32) in [0, 1].
            - cate_targets (tf.Tensor): [B, N] (int32).
            - mask_targets (tf.Tensor): [B, Hf, Wf, N] (uint8).
    """
    target_height, target_width = target_size
    augment_tf = tf.constant(augment)

    # Gather all shard paths (common suffixes)
    pattern = "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(train_tfrecord_directory, pattern))

    if not files:
        raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecord_directory}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    if number_images is not None:
        ds = ds.take(number_images)

    # Parse
    ds = ds.map(
        lambda x: parse_example(
            x, target_height=target_height, target_width=target_width, scale=scale, augment=augment_tf
        ),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic
    )

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


def create_coco_eval_dataset(
    train_tfrecord_directory: str,
    target_size: Tuple[int, int],
    batch_size: int,
    scale: float = 2.5,
    deterministic: bool = False,
    augment: bool = False,
    shuffle_buffer_size: Optional[int] = None,
    number_images: Optional[int] = None
) -> tf.data.Dataset:
    """
    Create a `tf.data.Dataset` from COCO TFRecord shards and emit Mask2Former targets.

    This utility scans a directory for `*.tfrecord` shards, builds a streaming `TFRecordDataset`,
    optionally shuffles and/or limits the number of examples, parses each example,
    constructs multi-scale Mask2Former targets, and batches/prefetches the dataset.

    Args:
        train_tfrecord_directory (str): Path to directory containing TFRecord shards.
        target_size (tuple): Target (height, width) for image & mask resizing.
        batch_size (int): Batch size for the resulting dataset.
        scale (float, optional): Feature-map scale factor used in target generation (e.g., 2.5).
            Defaults to 2.5.
        deterministic (bool, optional): If False, allow non-deterministic map parallelism.
            Defaults to False.
        augment (bool, optional): If True, apply data augmentations in `parse_example`.
            Defaults to True.
        shuffle_buffer_size (int, optional): Optional shuffle buffer size. If provided, shuffling is enabled.
            Defaults to None.
        number_images (int, optional): Optional cap on the number of images to take from the stream.
            Defaults to None.

    Returns:
        tf.data.Dataset: A dataset of batched tuples:
            - image_resized (tf.Tensor): [B, Ht, Wt, 3] (float32) in [0, 1].
            - cate_targets (tf.Tensor): [B, N] (int32).
            - mask_targets (tf.Tensor): [B, Hf, Wf, N] (uint8).
            - image_id (tf.Tensor): Image ID.
            - original_height (tf.Tensor): Original image height.
            - original_width (tf.Tensor): Original image width.
    """
    target_height, target_width = target_size
    augment_tf = tf.constant(augment)

    # Gather all shard paths (common suffixes)
    pattern = "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(train_tfrecord_directory, pattern))

    if not files:
        raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecord_directory}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    if number_images is not None:
        ds = ds.take(number_images)

    # Parse
    ds = ds.map(
        lambda x: parse_eval_example(
            x, target_height=target_height, target_width=target_width, scale=scale, augment=augment_tf
        ),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic
    )

    padded_shapes = (
        [target_size[0], target_size[1], 3],  # image shape
        [None, ],  # cate_target shape (num_instances,)
        [int(round(target_size[0] * scale)), int(round(target_size[1] * scale)), None],
        [], # image_id (scalar)
        [], # original_height (scalar)
        []  # original_width (scalar)
    )
    padding_values = (
        tf.constant(0.0, dtype=tf.float32),
        tf.constant(-1, dtype=tf.int32),
        tf.constant(0, dtype=tf.uint8),
        tf.constant(0, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
    )

    ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=True
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
