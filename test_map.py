"""
Author: Pavel Timonin
Created: 2026-01-17
Description: This script calculates Mean Average Precision (mAP) for the Mask2Former model on the validation dataset.
"""

import os
import json
import time
import tensorflow as tf
from config import Mask2FormerConfig
from coco_dataset_optimized import create_coco_tfrecord_dataset, COCOAnalysis
from model_functions import Mask2FormerModel


def create_label_map(coco_info):
    """
    Creates a mapping tensor from Dataset IDs (TFRecord values) to Model IDs.
    
    TFRecord contains `category_id - 1`.
    Ideally, if IDs are contiguous 1..N, then `category_id - 1` is 0..N-1.
    We create a map that supports the maximum ID found in the annotation.

    Args:
        coco_info (COCOAnalysis): Helper object with category info.

    Returns:
        tf.Tensor: Mapping tensor.
    """
    ids = coco_info.get_category_ids()
    if not ids:
        return tf.range(1, dtype=tf.int32)

    return tf.range(max(ids) + 1, dtype=tf.int32)


@tf.function(reduce_retracing=True)
def compute_iou(pred_masks, gt_masks):
    """
    Computes IoU between predicted masks and ground truth masks.

    Args:
        pred_masks (tf.Tensor): Predicted masks [N_pred, H, W] (0 or 1).
        gt_masks (tf.Tensor): Ground truth masks [N_gt, H, W] (0 or 1).

    Returns:
        tf.Tensor: IoU matrix [N_pred, N_gt].
    """
    # Flatten masks to [N, H*W]
    pred_flat = tf.reshape(pred_masks, (tf.shape(pred_masks)[0], -1))
    gt_flat = tf.reshape(gt_masks, (tf.shape(gt_masks)[0], -1))

    pred_flat = tf.cast(pred_flat, tf.float32)
    gt_flat = tf.cast(gt_flat, tf.float32)

    # Intersection: [N_pred, N_gt]
    intersection = tf.matmul(pred_flat, gt_flat, transpose_b=True)

    # Area of each mask
    area_pred = tf.reduce_sum(pred_flat, axis=1) # [N_pred]
    area_gt = tf.reduce_sum(gt_flat, axis=1) # [N_gt]

    # Union: area_pred + area_gt - intersection
    # Broadcast to [N_pred, N_gt]
    area_pred_bc = tf.expand_dims(area_pred, 1)
    area_gt_bc = tf.expand_dims(area_gt, 0)
    union = area_pred_bc + area_gt_bc - intersection

    # Avoid division by zero
    union = tf.maximum(union, 1e-6)

    return intersection / union

@tf.function(experimental_relax_shapes=True, reduce_retracing=True)
def process_single_image_predictions(
    pred_logits, pred_masks,
    gt_cate_dataset, gt_mask_dataset,
    target_height, target_width,
    label_map,
    score_threshold=0.0
):
    """
    Processes predictions for a single image to compute IoUs and matches.
    Run in Graph Mode.

    Args:
        pred_logits (tf.Tensor): [Q, num_classes+1]. Output logits.
                                 Index 0 is background (ignored).
                                 Indices 1..N are classes 0..N-1.
        pred_masks (tf.Tensor):  [Q, H_feat, W_feat]. Predicted masks.
        gt_cate_dataset (tf.Tensor): [K]. GT labels from dataset (Dataset IDs).
        gt_mask_dataset (tf.Tensor): [Hf, Wf, K]. GT masks from dataset.
        target_height (int): Target height for IoU calculation.
        target_width (int): Target width for IoU calculation.
        label_map (tf.Tensor): Label mapping tensor.
        score_threshold (float): Threshold to filter low-confidence predictions.

    Returns:
        tuple: (pred_scores, pred_labels, iou_matrix, gt_labels_mapped)
    """

    scores_fg = tf.nn.softmax(pred_logits, axis=-1)[:, 1:]

    # Get max score and class per query
    max_scores = tf.reduce_max(scores_fg, axis=-1)
    pred_labels = tf.argmax(scores_fg, axis=-1, output_type=tf.int32)

    # Filter by score threshold
    keep_indices = tf.where(max_scores > score_threshold)[:, 0]
    kept_scores = tf.gather(max_scores, keep_indices)
    kept_labels = tf.gather(pred_labels, keep_indices)
    kept_mask_logits = tf.gather(pred_masks, keep_indices) # [N_kept, Hf, Wf]

    kept_mask_logits_exp = tf.expand_dims(kept_mask_logits, -1)

    # Resize to target size. Using bilinear for logits is ok, then threshold.
    pred_masks_up = tf.image.resize(kept_mask_logits_exp, (target_height, target_width), method='bilinear')
    pred_masks_up = tf.squeeze(pred_masks_up, -1) # [N_kept, H, W]

    pred_masks_bin = tf.cast(pred_masks_up > 0.0, tf.float32)

    # Filter out empty targets (-1) from dataset
    valid_gt_indices = tf.where(gt_cate_dataset > -1)[:, 0]
    gt_cats_raw = tf.gather(gt_cate_dataset, valid_gt_indices)
    gt_labels_mapped = tf.gather(label_map, gt_cats_raw)

    valid_mapped_mask = gt_labels_mapped > -1
    gt_labels_final = tf.boolean_mask(gt_labels_mapped, valid_mapped_mask)
    valid_gt_indices_final = tf.boolean_mask(valid_gt_indices, valid_mapped_mask)

    gt_masks_grid = tf.gather(gt_mask_dataset, valid_gt_indices_final, axis=-1)
    n_gt_active = tf.shape(gt_masks_grid)[-1]

    def resize_gt_masks():
        gt_masks_grid_exp = tf.expand_dims(gt_masks_grid, 0) # [1, Hf, Wf, N_gt]
        up = tf.image.resize(gt_masks_grid_exp, (target_height, target_width), method='nearest')
        return tf.squeeze(up, 0) # [H, W, N_gt]

    def empty_gt_masks():
        return tf.zeros((target_height, target_width, 0), dtype=gt_masks_grid.dtype)

    gt_masks_up = tf.cond(n_gt_active > 0, resize_gt_masks, empty_gt_masks)

    gt_masks_up = tf.transpose(gt_masks_up, perm=[2, 0, 1]) # [N_gt, H, W]

    gt_masks_bin = tf.cast(tf.cast(gt_masks_up, tf.float32) > 0.5, tf.float32)

    n_pred = tf.shape(kept_scores)[0]
    n_gt = tf.shape(gt_labels_final)[0]

    if n_pred == 0 or n_gt == 0:
        return kept_scores, kept_labels, tf.zeros((n_pred, n_gt)), gt_labels_final

    iou_mat = compute_iou(pred_masks_bin, gt_masks_bin) # [N_pred, N_gt]

    return kept_scores, kept_labels, iou_mat, gt_labels_final


@tf.function
def compute_matches_single_tf(iou_matrix, iou_thresh):
    """
    Greedy matching of predictions to ground truth for a single threshold.

    Args:
        iou_matrix (tf.Tensor): [N_dt, N_gt] float32. Rows are predictions (sorted), cols are GT.
        iou_thresh (tf.Tensor): Scalar float IoU threshold.

    Returns:
        tf.Tensor: dt_matched [N_dt] bool.
    """
    n_dt = tf.shape(iou_matrix)[0]
    n_gt = tf.shape(iou_matrix)[1]

    gt_covered = tf.zeros(n_gt, dtype=tf.bool)
    dt_matched = tf.zeros(n_dt, dtype=tf.bool)

    if n_gt == 0 or n_dt == 0:
        return dt_matched

    def body(i, gt_cov, dt_mat):
        row = iou_matrix[i]
        valid_mask = tf.logical_and(row >= iou_thresh, tf.logical_not(gt_cov))
        has_valid = tf.reduce_any(valid_mask)

        def match_found():
            # Find Best IoU among valid
            # Mask invalid with -1
            masked_row = tf.where(valid_mask, row, -1.0)
            best_idx = tf.argmax(masked_row, output_type=tf.int32)

            # Update gt_covered[best_idx] = True
            new_gt_cov = tf.tensor_scatter_nd_update(gt_cov, [[best_idx]], [True])

            # Update dt_matched[i] = True
            new_dt_mat = tf.tensor_scatter_nd_update(dt_mat, [[i]], [True])
            return new_gt_cov, new_dt_mat

        def no_match():
            return gt_cov, dt_mat

        new_gc, new_dm = tf.cond(has_valid, match_found, no_match)
        return i + 1, new_gc, new_dm

    # Loop over all detections
    _, _, dt_matched_final = tf.while_loop(
        lambda i, g, d: i < n_dt,
        body,
        [0, gt_covered, dt_matched],
        parallel_iterations=1,
    )

    return dt_matched_final


@tf.function
def match_predictions_tf(iou_matrix, iou_thresholds):
    """
    Greedy matching of predictions to ground truth for multiple thresholds.

    Args:
        iou_matrix (tf.Tensor): [N_dt, N_gt] float32. Rows are predictions (sorted), cols are GT.
        iou_thresholds (tf.Tensor): [N_thresh] float32. IoU thresholds.

    Returns:
        tf.Tensor: dt_matched [N_thresh, N_dt] bool.
    """
    # Use map_fn to process all thresholds in the graph
    # Results in [N_thresh, N_dt]
    return tf.map_fn(
        lambda thresh: compute_matches_single_tf(iou_matrix, thresh),
        iou_thresholds,
        fn_output_signature=tf.bool,
    )


class MAPEvaluator:
    """
    TensorFlow-optimized mAP Evaluator.

    Accumulates results in lists of Tensors, computes mAP using TF ops.

    Args:
        num_classes (int): Number of classes.
        iou_thresholds (list or tf.Tensor, optional): IoU thresholds.
            Defaults to 0.5:0.05:0.95.
    """

    def __init__(self, num_classes, iou_thresholds=None):
        self.num_classes = num_classes
        if iou_thresholds is not None:
             self.iou_thresholds = iou_thresholds
        else:
             self.iou_thresholds = tf.linspace(0.5, 0.95, 10)

        # Convert thresholds to tensor explicitly if not already
        if not tf.is_tensor(self.iou_thresholds):
            self.iou_thresholds = tf.convert_to_tensor(
                self.iou_thresholds, dtype=tf.float32
            )

        self.stats = {}
        for c in range(num_classes):
            self.stats[c] = {
                "scores": [],
                "matches": [],  # Changed from 'tp' dict to single list of N_thresh x N_dt matrices
                "n_gt": 0,
            }

    def update(self, pred_scores, pred_labels, iou_matrix, gt_labels):
        """
        Update stats with results from a single image.

        Args:
            pred_scores (tf.Tensor): [N_dt] Predicted scores.
            pred_labels (tf.Tensor): [N_dt] Predicted labels.
            iou_matrix (tf.Tensor): [N_dt, N_gt] IoU matrix.
            gt_labels (tf.Tensor): [N_gt] Ground truth labels.
        """
        all_labels = tf.concat([tf.cast(pred_labels, tf.int32), gt_labels], axis=0)
        unique_classes, _ = tf.unique(all_labels)
        unique_classes = unique_classes.numpy()

        for cls_id in unique_classes:
            cls_id = int(cls_id)
            if cls_id not in self.stats:
                continue

            gt_indices = tf.where(tf.equal(gt_labels, cls_id))[:, 0]
            dt_indices = tf.where(tf.equal(pred_labels, cls_id))[:, 0]

            n_gt_cls = tf.shape(gt_indices)[0].numpy()
            self.stats[cls_id]["n_gt"] += n_gt_cls

            if tf.shape(dt_indices)[0] == 0:
                continue

            # Gather data for this class
            # IoU matrix sub-block: [N_dt_cls, N_gt_cls]
            cls_iou_mat = tf.gather(iou_matrix, dt_indices)  # rows
            if n_gt_cls > 0:
                cls_iou_mat = tf.gather(cls_iou_mat, gt_indices, axis=1)  # cols
            else:
                cls_iou_mat = tf.zeros((tf.shape(dt_indices)[0], 0), dtype=tf.float32)

            cls_scores = tf.gather(pred_scores, dt_indices)

            # Sort by score desc
            sort_idx = tf.argsort(cls_scores, direction="DESCENDING")
            cls_scores_sorted = tf.gather(cls_scores, sort_idx)
            cls_iou_mat_sorted = tf.gather(cls_iou_mat, sort_idx)

            self.stats[cls_id]["scores"].append(cls_scores_sorted)

            # Compute matches for all thresholds at once
            if n_gt_cls == 0:
                num_thresh = tf.shape(self.iou_thresholds)[0]
                num_dt = tf.shape(dt_indices)[0]
                matches = tf.zeros((num_thresh, num_dt), dtype=tf.bool)
            else:
                matches = match_predictions_tf(
                    cls_iou_mat_sorted, self.iou_thresholds
                )

            self.stats[cls_id]["matches"].append(matches)

    @tf.function(reduce_retracing=True)
    def _compute_ap_per_class(self, tp_cat, n_gt):
        """
        Compute Average Precision for a single class and method.

        Args:
            tp_cat (tf.Tensor): [N_total_dt] bool. True if matched.
            n_gt (tf.Tensor): Scalar int. Total ground truth count.

        Returns:
            tf.Tensor: AP scalar.
        """
        n_gt = tf.cast(n_gt, tf.float32)
        if n_gt == 0:
            return 0.0

        # In this sorted list (globally sorted), compute cumsum
        tp = tf.cast(tp_cat, tf.float32)
        fp = 1.0 - tp

        tp_sum = tf.cumsum(tp)
        fp_sum = tf.cumsum(fp)

        # Precision and Recall
        recalls = tp_sum / n_gt
        precisions = tp_sum / (tp_sum + fp_sum + 1e-6)

        # Monotonic decreasing precisions (accumulate max from right)
        # p_i = max(p_i, p_{i+1}, ...)
        # Scan reverse with max
        precisions = tf.scan(lambda a, x: tf.maximum(a, x), precisions, reverse=True)

        # COCO 101-point method
        recall_steps = tf.linspace(0.0, 1.0, 101)

        # Find indices where recalls >= step
        indices = tf.searchsorted(recalls, recall_steps, side="left")

        # Gather precisions at these indices
        N = tf.shape(precisions)[0]
        indices_clipped = tf.minimum(indices, N - 1)

        gathered_precisions = tf.gather(precisions, indices_clipped)

        # If the found index is N (meaning step > max_recall), the precision is 0.
        mask = indices < N
        gathered_precisions = tf.where(mask, gathered_precisions, 0.0)

        return tf.reduce_mean(gathered_precisions)

    def calculate_map(self):
        """
        Compute mAP across all classes.

        Returns:
            float: mAP score.
        """
        aps = []
        print("Computing mAP...")
        for cls_id in self.stats:
            n_gt = self.stats[cls_id]["n_gt"]
            if n_gt == 0:
                continue

            # Global sort for this class
            scores_list = self.stats[cls_id]["scores"]
            if not scores_list:
                aps.append(0.0)
                continue

            all_scores = tf.concat(scores_list, axis=0)
            global_sort_idx = tf.argsort(all_scores, direction="DESCENDING")

            # Collect matches: List of [N_thresh, N_dt_batch] -> Concat -> [N_thresh, Total_DT]
            matches_list = self.stats[cls_id]["matches"]
            all_matches_matrix = tf.concat(matches_list, axis=1)

            all_matches_sorted = tf.gather(all_matches_matrix, global_sort_idx, axis=1)

            # Calculate AP for each threshold row
            num_thresh = tf.shape(self.iou_thresholds)[0]
            ap_thresholds = tf.TensorArray(tf.float32, size=num_thresh)

            for t in range(num_thresh):
                row = all_matches_sorted[t]
                ap = self._compute_ap_per_class(row, tf.constant(n_gt))
                ap_thresholds = ap_thresholds.write(t, ap)

            ap_thresholds_stacked = ap_thresholds.stack()

            # Mean for this class over simple average of thresholds
            cls_mAP = tf.reduce_mean(ap_thresholds_stacked)
            aps.append(cls_mAP)

        if not aps:
            return 0.0

        return tf.reduce_mean(tf.stack(aps)).numpy()



def main():
    """
    Main function to run mAP evaluation.

    Loads the validation dataset and model, runs inference, and computes mAP.
    """
    cfg = Mask2FormerConfig()

    print(f"Loading test dataset from: {cfg.tfrecord_test_path}")

    dataset = create_coco_tfrecord_dataset(
        train_tfrecord_directory=cfg.tfrecord_test_path,
        target_size=(cfg.img_height, cfg.img_width),
        batch_size=1,
        scale=cfg.image_scales[0],
        deterministic=True,
        augment=False,
        shuffle_buffer_size=None,
        number_images=None
    )

    coco_info = COCOAnalysis(cfg.train_annotation_path)
    num_classes = coco_info.get_num_classes()
    print(f"Number of classes detected: {num_classes}")

    model = Mask2FormerModel(
        input_shape=(cfg.img_height, cfg.img_width, 3),
        transformer_input_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024
    )
    dummy_input = tf.zeros((1, cfg.img_height, cfg.img_width, 3))
    model(dummy_input)

    checkpoint_dir = cfg.test_model_path
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Restored model from {manager.latest_checkpoint}")
    else:
        print("No checkpoint found! Running with random weights (mAP will be garbage).")

    evaluator = MAPEvaluator(num_classes)
    score_thresh = 0.05
    label_map = create_label_map(coco_info)

    start_time = time.time()

    @tf.function
    def predict_step(img):
        return model(img, training=False)

    print("Starting evaluation...")
    for i, (image, cate_target, mask_target) in enumerate(dataset):
        pred_logits, pred_masks, _ = predict_step(image)

        s, l, iou, gl = process_single_image_predictions(
            pred_logits[0], pred_masks[0],
            cate_target[0], mask_target[0],
            target_height=cfg.img_height,
            target_width=cfg.img_width,
            label_map=label_map,
            score_threshold=score_thresh
        )

        evaluator.update(s, l, iou, gl)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images...")

    total_time = time.time() - start_time
    print(f"Evaluation finished in {total_time:.2f}s")

    mean_ap = evaluator.calculate_map()
    print(f"mAP: {mean_ap:.4f}")

if __name__ == '__main__':
    main()