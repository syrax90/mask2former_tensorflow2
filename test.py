"""
Author: Pavel Timonin
Created: 2025-09-28
Description: This script performs the model testing process.
"""

import cv2
import os
import random

import numpy as np
import tensorflow as tf


from config import Mask2FormerConfig
from coco_dataset_optimized import COCOAnalysis
from model_functions import Mask2FormerModel
import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def preprocess_image(image_path, input_size=(320, 320)):
    """
    Load an image, resize to a fixed size, and normalize pixel values to [0, 1].

    Also returns the original image shape and the unnormalized RGB image.

    Args:
        image_path (str): Path to the input image file.
        input_size (tuple, optional): Target size `(width, height)` to resize the image to.
            Defaults to `(320, 320)`.

    Returns:
        tuple: A 3-tuple `(img_resized, original_shape, img)` where:
            - `img_resized` (np.ndarray): Resized and normalized RGB image of shape `(H, W, 3)`,
              dtype `float32`, values in `[0, 1]`.
            - `original_shape` (tuple): Original image height and width as `(H_orig, W_orig)`.
            - `img` (np.ndarray): Original RGB image **before** resizing and normalization,
              dtype `uint8`, values in `[0, 255]`.

    Raises:
        ValueError: If the image cannot be read from `image_path`.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]  # (H, W)

    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32)
    # Note: preprocess_input assumes RGB and might zero-center the data (not necessarily [0,1]).
    img_resized = tf.keras.applications.resnet50.preprocess_input(img_resized)

    return img_resized, original_shape, img

@tf.function
def get_instances(cate_preds, mask_preds, score_threshold):
    """
    Convert model head outputs into per-instance masks/scores/classes.

    This function gathers all grid-cell/class pairs whose classification
    probability exceeds `score_threshold`. It returns the corresponding
    mask logits in their original low resolution (without upsampling).

    Args:
        cate_preds (tf.Tensor): Category logits of shape `[1, sum(S_i*S_i), C]` where `C` includes background at index 0.
        mask_preds (tf.Tensor): Mask kernels of shape `[1, H_l, W_l, sum(S_i*S_i)]`.
        score_threshold (float): Minimum class probability to keep a (cell, class) candidate.

    Returns:
        tuple: A tuple containing:
            - selected_masks (tf.Tensor): Shape `(K, H_l, W_l)` with values in `[0, 1]` (after sigmoid).
            - selected_scores (tf.Tensor): Shape `(K,)` with per-instance class probabilities.
            - selected_classes (tf.Tensor): Shape `(K,)` with zero-based foreground class indices.
            - selected_cells (tf.Tensor): Shape `(K,)` with original cell indices (useful for debugging).

    Notes:
        * If no candidate passes the threshold, returns empty tensors.
        * This function assumes background occupies class index `0` in `cate_preds` and removes it for thresholding.
    """

    cate_out = cate_preds[0]  # => shape [sum(S_i*S_i), num_classes]

    mask_out = mask_preds[0]  # => shape (H_l, W_l, S_l^2)
    mask_out = tf.transpose(mask_out, perm=[1, 2, 0])  # [H_l, W_l, Q]

    cate_prob = tf.nn.softmax(cate_out, axis=-1)  # shape [sum(S_i*S_i), num_classes]

    mask_prob = tf.sigmoid(mask_out)  # shape (H_l, W_l, S_l^2)

    # Get rid of the background. Assume background has 0th index
    cate_prob = cate_prob[..., 1:]

    mask_bool = cate_prob >= tf.cast(score_threshold, tf.float32)  # shape (S_l*S_l, C)

    idx = tf.where(mask_bool)

    if tf.shape(idx)[0] == 0:
        H_l = tf.shape(mask_out)[0]
        W_l = tf.shape(mask_out)[1]
        return (tf.zeros([0, H_l, W_l], tf.float32),
                tf.zeros([0], tf.float32),
                tf.zeros([0], tf.int32),
                tf.zeros([0], tf.int32))

    selected_scores = tf.gather_nd(cate_prob, idx)  # [K]
    selected_classes = tf.cast(idx[:, 1], tf.int32)  # [K]
    selected_cells = tf.cast(idx[:, 0], tf.int32)  # [K]
    # mask_prob is [H_l, W_l, Q], we want to gather along axis -1 using selected_cells
    masks = tf.gather(mask_prob, selected_cells, axis=-1)  # [H_l, W_l, K]
    masks = tf.transpose(masks, [2, 0, 1])  # [K, H_l, W_l]

    return masks, selected_scores, selected_classes, selected_cells


def postprocess_model_outputs(
    cate_preds,          # Tensor of shape [1, sum(S_i*S_i), C]
    mask_preds,          # Tensor of shape [1, H_l, W_l, sum(S_i*S_i)]
    resized_image_shape,  # (resized_h, resized_w)
    score_threshold=0.5
):
    """
    Convert raw model outputs into final instance predictions.

    Runs candidate extraction (`get_instances`) and upsamples the masks.

    Args:
        cate_preds (tf.Tensor): Category logits of shape `[1, sum(S_i*S_i), C]`.
        mask_preds (tf.Tensor): Mask kernels of shape `[1, H_l, W_l, sum(S_i*S_i)]`.
        resized_image_shape (tuple): `(resized_h, resized_w)` used for upsampling masks.
        score_threshold (float, optional): Classification probability threshold for candidate selection.
            Defaults to `0.5`.

    Returns:
        list: A sorted list of instance dictionaries. Each item has keys:
            - `"class_id"` (int): Zero-based class index (without background).
            - `"score"` (float): Final score.
            - `"mask"` (np.ndarray): Binary mask of shape `(resized_h, resized_w)`,
              `dtype=uint8` with values in `{0,1}`.
    """
    resized_h, resized_w = resized_image_shape

    all_masks_low_res, all_scores, all_classes, _ = get_instances(
        cate_preds, mask_preds, score_threshold
    )

    if tf.shape(all_masks_low_res)[0] == 0:
        return []

    sorted_indices = tf.argsort(all_scores, direction='DESCENDING')
    final_scores = tf.gather(all_scores, sorted_indices)
    final_classes = tf.gather(all_classes, sorted_indices)
    final_masks_low = tf.gather(all_masks_low_res, sorted_indices)

    # Upsample masks ONLY at the end
    masks_expanded = final_masks_low[..., tf.newaxis]

    masks_up = tf.image.resize(
        masks_expanded, [resized_h, resized_w],
        method='bilinear', antialias=True
    )  # [K, resized_h, resized_w, 1]

    masks_up = tf.squeeze(masks_up, axis=-1)  # [K, resized_h, resized_w]
    final_masks_bool = (masks_up > 0.5)

    final_masks_np = final_masks_bool.numpy().astype(np.uint8)
    final_scores_np = final_scores.numpy()
    final_classes_np = final_classes.numpy()

    instances = []
    for i in range(final_masks_np.shape[0]):
        instances.append({
            "class_id": int(final_classes_np[i]),
            "score": float(final_scores_np[i]),
            "mask": final_masks_np[i]
        })

    return instances


def draw_instance_masks(
    original_image: np.ndarray,
    instances: list,
    show_labels=True,
    class_names=None
):
    """
    Overlay instance masks and optional labels on an image.

    Applies semi-transparent color overlays for each instance mask and, if
    requested, draws a class label near the first pixel of the mask.

    Args:
        original_image (np.ndarray): Input image on which to draw. Typically **BGR** (OpenCV convention),
            `dtype=uint8`, shape `(H, W, 3)`.
        instances (list): List of instance dicts as returned by `postprocess_model_outputs` with keys:
            `class_id`, `score`, and `mask` (binary `(h, w)` array).
        show_labels (bool, optional): Whether to render text labels. Defaults to `True`.
        class_names (dict, optional): Mapping from class id to readable name. If provided and indexable by
            `class_id`, the corresponding name is used for the label. Defaults to None.

    Returns:
        np.ndarray: Annotated image (same shape and dtype as `original_image`).

    Notes:
        * Overlay color is randomly sampled for each instance.
        * If `class_names` is missing a key, a fallback `"ID=<id>, <score>"` label is used.
    """
    vis_image = original_image.copy()
    orig_h, orig_w = vis_image.shape[:2]

    for inst in instances:
        mask_resized = inst["mask"]
        mask_orig = cv2.resize(
            mask_resized.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        # mask_orig => shape (orig_h, orig_w) in {0,1}
        # Color overlay
        color = [random.randint(0, 255) for _ in range(3)]
        alpha = 0.5

        vis_image[mask_orig == 1] = (
            alpha * np.array(color) + (1 - alpha) * vis_image[mask_orig == 1]
        )

        ys, xs = np.where(mask_orig == 1)
        if len(ys) > 0:
            y0, x0 = ys[0], xs[0]
            score_str = f"{inst['score']:.2f}"
            if class_names and (0 <= inst['class_id'] < len(class_names)):
                label_str = f"{class_names.get(inst['class_id'])}"#: {score_str}"
            else:
                label_str = f"ID={inst['class_id']}, {score_str}"
            cv2.putText(
                vis_image, label_str if show_labels == True else "",
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

    return vis_image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    cfg = Mask2FormerConfig()

    model_path = cfg.test_model_path
    input_shape = (cfg.img_height, cfg.img_width, 3)
    coco_info = COCOAnalysis(cfg.train_annotation_path)
    categories = coco_info.categories
    coco_classes = {cat['id'] - 1: cat['name'] for cat in categories}

    coco_info = COCOAnalysis(cfg.train_annotation_path)
    num_classes = coco_info.get_num_classes()
    img_height, img_width = cfg.img_height, cfg.img_width
    model = Mask2FormerModel(
        input_shape=(img_height, img_width, 3),
        transformer_input_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024
    )
    model.build((None, img_height, img_width, 3))

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = model_path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()
        print(f"Restored from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Load previous model is True but no checkpoint found in {model_path}")

    if not os.path.exists('images/res/'): os.mkdir('images/res/')
    path_dir = os.listdir('images/test')

    for k, filename in enumerate(path_dir):
        image_path = f'images/test/{filename}'
        img_resized, (orig_h, orig_w), img_rgb = preprocess_image(image_path, input_size=input_shape[:2])

        img_batch = np.expand_dims(img_resized, axis=0)
        img_batch = tf.convert_to_tensor(img_batch)

        class_outputs, mask_outputs, _ = model.predict(img_batch)

        all_instances = []
        instances = postprocess_model_outputs(
            cate_preds=class_outputs,
            mask_preds=mask_outputs,
            resized_image_shape=input_shape[:2],
            score_threshold=cfg.score_threshold
        )

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        annotated_img = draw_instance_masks(img_bgr, instances, show_labels=True, class_names=coco_classes)

        output_filename = os.path.basename(image_path)
        save_path = os.path.join('images/res', output_filename)
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved annotated image: {save_path}")

    exit(0)