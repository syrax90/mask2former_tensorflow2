"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script draws dataset with masks by categories to understand what data we fit to the model.
"""


import os

import tensorflow as tf
import numpy as np
import cv2
import random

from config import Mask2FormerConfig
from coco_dataset_optimized import create_coco_tfrecord_dataset, COCOAnalysis
import shutil

def draw_instance_predictions(
    image,
    cate_target,
    mask_target,
    class_names=None,
    show_labels=True
):
    """
    Returns a BGR uint8 visualization with colored masks and optional class labels.

    If the input image contains negative values, it is assumed to be zero-centered
    ResNet50 preprocessed data (BGR). It will be denormalized by adding the mean.
    Otherwise, it is treated as an RGB image in [0, 1] or [0, 255].

    Coloring, transparency and blending follow the behavior in draw_instance_masks():
      - random color per instance
      - alpha = 0.5
      - vis[mask==1] = alpha*color + (1-alpha)*vis[mask==1]
      - label near the first pixel of the mask

    Args:
        image (tf.Tensor or np.ndarray): Input image of shape (H, W, 3).
            Can be:
            1. RGB in [0, 1] or [0, 255]
            2. BGR zero-centered (ResNet preprocessed)
        cate_target (tf.Tensor or np.ndarray): Category targets of shape [sum(S_i^2)].
            Values are int32, -1 for empty, otherwise category_id.
        mask_target (tf.Tensor or np.ndarray): Mask targets of shape [Hf, Wf, sum(S_i^2)].
            Values are uint8 {0, 1}.
        class_names (dict, optional): Dictionary mapping category_id to class name string.
            Defaults to None.
        show_labels (bool, optional): Whether to show text labels on the image.
            Defaults to True.

    Returns:
        np.ndarray: The visualized image as a BGR uint8 array.
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(cate_target, tf.Tensor):
        cate_target = cate_target.numpy()
    if isinstance(mask_target, tf.Tensor):
        mask_target = mask_target.numpy()

    img = image.copy()

    # Check for ResNet preprocessing (zero-centered, BGR, often negative values)
    if np.min(img) < 0:
        # Restore BGR image from zero-centered
        # Mean values for ResNet: [103.939, 116.779, 123.68]
        img[..., 0] += 103.939
        img[..., 1] += 116.779
        img[..., 2] += 123.68
        img = np.clip(img, 0, 255).astype(np.uint8)
        # It is already BGR, so we use it directly
        vis = img
    else:
        # Standard RGB image handling
        if img.dtype != np.uint8:
            # Assume [0, 1] float if max <= 1.5 (safety margin)
            if img.max() <= 1.5:
                 img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                 img = np.clip(img, 0, 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    H, W = vis.shape[:2]
    Hf, Wf, C = mask_target.shape
    assert C == cate_target.shape[0], "mask_target channels and cate_target length must match"

    # Upsample all masks to image size using NEAREST (binary kept)
    if (Hf != H) or (Wf != W):
        up_masks = np.zeros((H, W, C), dtype=np.uint8)
        for c in range(C):
            up_masks[..., c] = cv2.resize(mask_target[..., c], (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        up_masks = mask_target

    # Positive channels (instances)
    pos_idx = np.where(cate_target >= 0)[0]
    if pos_idx.size == 0:
        return vis

    alpha = 0.5

    for ch in pos_idx:
        class_id = int(cate_target[ch])
        mask_bin = (up_masks[..., ch] > 0)

        if mask_bin.sum() == 0:
            continue

        color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)

        # Per-pixel alpha blend exactly like:
        # vis[mask==1] = alpha*color + (1-alpha)*vis[mask==1]
        m = mask_bin
        if m.any():
            region = vis[m].astype(np.float32)
            vis[m] = (alpha * color + (1.0 - alpha) * region).astype(np.uint8)

        # Label near first pixel in the mask (if enabled)
        if show_labels:
            ys, xs = np.where(m)
            if len(ys) > 0:
                y0, x0 = int(ys[0]), int(xs[0])
                if isinstance(class_names, dict) and (class_id in class_names):
                    label_str = f"{class_names.get(class_id)}"
                else:
                    label_str = f"ID={class_id}"
                cv2.putText(
                    vis, label_str,
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

    return vis

def save_dataset_preview(dataset, coco_classes, out_dir, max_images=50, show_labels=True):
    """
    Save a grid-free preview of Mask2Former targets for a dataset.

    Iterates over a `tf.data.Dataset` that yields `(images, cate_targets, mask_targets)`
    batches, renders each sample with `draw_instance_predictions`, and writes PNG files to
    `out_dir` until `max_images` previews are saved.

    Args:
        dataset (tf.data.Dataset): A dataset yielding batches of:
            - images: Tensor of shape (B, H, W, 3), RGB.
            - cate_targets: Tensor of shape (B, C), int32.
            - mask_targets: Tensor of shape (B, Hf, Wf, C), uint8.
        coco_classes (dict): Mapping of {category_id (int): class name (str)}.
        out_dir (str): Destination directory to store rendered PNG files.
        max_images (int, optional): Maximum number of samples to save. Defaults to 50.
        show_labels (bool, optional): If True, overlay class labels. Defaults to True.
    """
    import os, cv2
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for batch in dataset:
        images, cate_targets, mask_targets = batch
        bs = images.shape[0]
        for b in range(bs):
            vis = draw_instance_predictions(
                images[b].numpy(),
                cate_targets[b].numpy(),
                mask_targets[b].numpy(),
                class_names=coco_classes,
                show_labels=show_labels
            )
            cv2.imwrite(os.path.join(out_dir, f"sample_{saved:04d}.png"), vis)
            saved += 1
            if saved >= max_images:
                return

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    cfg = Mask2FormerConfig()
    coco_info = COCOAnalysis(cfg.train_annotation_path)
    categories = coco_info.categories
    coco_classes = {cat['id'] - 1: cat['name'] for cat in categories}

    num_classes = len(coco_classes)
    img_height, img_width = cfg.img_height, cfg.img_width

    ds = create_coco_tfrecord_dataset(
        train_tfrecord_directory=cfg.tfrecord_dataset_directory_path,
        target_size=(img_height, img_width),
        batch_size=cfg.batch_size,
        scale=cfg.image_scales[0],
        augment=cfg.augment,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        number_images=cfg.number_images)

    out_dir = 'images/dataset_test'
    shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)
    save_dataset_preview(ds, coco_classes, out_dir, max_images=200)  # adjust as needed
    print(f"Saved previews to: {out_dir}")