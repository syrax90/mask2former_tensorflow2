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
from coco_dataset_optimized import create_coco_tfrecord_dataset, get_classes
from model_functions import Mask2FormerModel


def get_dataset_to_model_map_from_json(json_path, num_classes, classes_path):
    """
    Creates a mapping tensor from Dataset IDs to Model IDs.

    Args:
        json_path (str): Unused. Kept for compatibility.
        num_classes (int): Number of classes.
        classes_path (str): Unused. Kept for compatibility.

    Returns:
        tf.Tensor: Identity mapping tensor of shape [num_classes + safety_buffer].
    """
    # Assume dataset IDs are already 0..num_classes-1 and match model classes.
    # We provide a large enough identity map just in case.
    # Size: num_classes + some buffer to avoid OOB on unexpected inputs (though they should be filtered)
    map_size = num_classes + 1
    return tf.range(map_size, dtype=tf.int32)


# Function to compute Intersection over Union (IoU)
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

    # Cast to float32 for matmul
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

    # --- 1. Process Predictions ---
    # Model Output: [Background, Class0, Class1, ...]
    # We want to keep only Foreground classes.
    # Scores: Discard index 0 (Background). Keep 1..end.
    scores_fg = tf.nn.softmax(pred_logits, axis=-1)[:, 1:] # [Q, num_classes] (Indices 1..N are mapped to 0..N-1)

    # Get max score and class per query
    max_scores = tf.reduce_max(scores_fg, axis=-1)
    # argmax gives index in generic [0..79] space, which is what we want
    pred_labels = tf.argmax(scores_fg, axis=-1, output_type=tf.int32)

    # Filter by score threshold
    keep_indices = tf.where(max_scores > score_threshold)[:, 0]

    kept_scores = tf.gather(max_scores, keep_indices)
    kept_labels = tf.gather(pred_labels, keep_indices)
    kept_mask_logits = tf.gather(pred_masks, keep_indices) # [N_kept, Hf, Wf]

    # Upsample predicted masks to image size
    # Expand dims for resize: [N_kept, Hf, Wf, 1]
    kept_mask_logits_exp = tf.expand_dims(kept_mask_logits, -1)

    # Resize to target size. Using bilinear for logits is ok, then threshold.
    pred_masks_up = tf.image.resize(kept_mask_logits_exp, (target_height, target_width), method='bilinear')
    pred_masks_up = tf.squeeze(pred_masks_up, -1) # [N_kept, H, W]

    # Binarize (logits > 0)
    pred_masks_bin = tf.cast(pred_masks_up > 0.0, tf.float32)

    # --- 2. Process Ground Truth ---
    # Filter out empty targets (-1) from dataset
    valid_gt_indices = tf.where(gt_cate_dataset > -1)[:, 0]

    gt_cats_raw = tf.gather(gt_cate_dataset, valid_gt_indices)

    # MAP Dataset IDs to Model IDs
    # Safely gather from map
    # Check bounds first? If generic dataset, IDs could be anything.
    # Fallback to identify if map is small? No, dynamic map should be big enough or identity.
    # We assume label_map covers the range.

    gt_labels_mapped = tf.gather(label_map, gt_cats_raw) # [N_gt]

    # Filter out GTs that mapped to -1 (invalid classes, e.g. "street sign" / missing IDs)
    # This shouldn't happen if dataset is clean COCO, but safe to handle.
    valid_mapped_mask = gt_labels_mapped > -1
    gt_labels_final = tf.boolean_mask(gt_labels_mapped, valid_mapped_mask)
    valid_gt_indices_final = tf.boolean_mask(valid_gt_indices, valid_mapped_mask)

    # Fetch corresponding masks
    gt_masks_grid = tf.gather(gt_mask_dataset, valid_gt_indices_final, axis=-1) # [Hf, Wf, N_gt]

    # Check if we have any GT masks to resize
    n_gt_active = tf.shape(gt_masks_grid)[-1]

    def resize_gt_masks():
        gt_masks_grid_exp = tf.expand_dims(gt_masks_grid, 0) # [1, Hf, Wf, N_gt]
        up = tf.image.resize(gt_masks_grid_exp, (target_height, target_width), method='nearest')
        return tf.squeeze(up, 0) # [H, W, N_gt]

    def empty_gt_masks():
        return tf.zeros((target_height, target_width, 0), dtype=gt_masks_grid.dtype)

    # If N_gt > 0, resize. Else return empty.
    gt_masks_up = tf.cond(n_gt_active > 0, resize_gt_masks, empty_gt_masks)

    gt_masks_up = tf.transpose(gt_masks_up, perm=[2, 0, 1]) # [N_gt, H, W]

    gt_masks_bin = tf.cast(tf.cast(gt_masks_up, tf.float32) > 0.5, tf.float32)

    # --- 3. Compute IoU ---
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
        # i: current dt index (already sorted by score)
        row = iou_matrix[i]  # [N_gt]

        # Mask out covered GTs and check threshold
        # valid: iou >= thresh AND not covered
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
        parallel_iterations=1,  # Serial execution required for greedy matching
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

        # Storage:
        # { cls_id: {
        #      'scores': [list_of_tensors],
        #      'matches': [list_of_tensors (N_thresh, N_dt)],
        #      'n_gt': int
        #   }
        # }
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
        # Logic partially in Python to manage the Class dictionary structure,
        # but heavy lifting in TF.

        # 1. Get unique classes in this image
        # Using TF for uniqueness
        all_labels = tf.concat([tf.cast(pred_labels, tf.int32), gt_labels], axis=0)
        unique_classes, _ = tf.unique(all_labels)
        unique_classes = unique_classes.numpy()  # iterate in python

        for cls_id in unique_classes:
            cls_id = int(cls_id)
            if cls_id not in self.stats:
                continue

            # tf.gather indices
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

            # Store scores
            self.stats[cls_id]["scores"].append(cls_scores_sorted)

            # Compute matches for all thresholds at once
            if n_gt_cls == 0:
                # [N_thresh, N_dt] false
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

            all_scores = tf.concat(scores_list, axis=0)  # [Total_DT]

            # Sort globally
            global_sort_idx = tf.argsort(all_scores, direction="DESCENDING")

            # Collect matches: List of [N_thresh, N_dt_batch] -> Concat -> [N_thresh, Total_DT]
            matches_list = self.stats[cls_id]["matches"]
            # Concat along axis 1 (time/detection axis)
            all_matches_matrix = tf.concat(matches_list, axis=1) # [N_thresh, Total_DT]

            # Compute AP per threshold
            # We can iterate over thresholds in Python (only 10 usually)
            # but vectorized gathering is better.

            # Reorder matches according to global sort
            # global_sort_idx is [Total_DT]
            # We want to shuffle the columns of all_matches_matrix
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
    # 1. Config
    cfg = Mask2FormerConfig()

    # 2. Dataset
    # Using 'test' TFRecords path as configured
    print(f"Loading test dataset from: {cfg.tfrecord_test_path}")

    # Note: create_coco_tfrecord_dataset handles parsing
    # We disable shuffle/augment for validation
    dataset = create_coco_tfrecord_dataset(
        train_tfrecord_directory=cfg.tfrecord_test_path,
        target_size=(cfg.img_height, cfg.img_width),
        batch_size=1, # Evaluating one by one is simpler for mAP matching
        scale=cfg.image_scales[0],
        deterministic=True,
        augment=False,
        shuffle_buffer_size=None,
        number_images=None # Process all
    )

    # 3. Model
    # Get classes to determine count (should be 80 for COCO)
    # If file missing, default to 80
    if os.path.exists(cfg.classes_path):
        class_names = get_classes(cfg.classes_path)
        num_classes = len(class_names)
    else:
        print(f"Classes file {cfg.classes_path} not found. Defaulting to 80 classes.")
        num_classes = 80

    model = Mask2FormerModel(
        input_shape=(cfg.img_height, cfg.img_width, 3),
        transformer_input_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024
    )
    # Build model
    dummy_input = tf.zeros((1, cfg.img_height, cfg.img_width, 3))
    model(dummy_input)

    # Load weights
    checkpoint_dir = cfg.test_model_path # Or model_path
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

    # 4. Evaluation Loop
    evaluator = MAPEvaluator(num_classes)

    score_thresh = 0.05 # Optimization: ignore very low confidence predictions

    # Init label map
    label_map = get_dataset_to_model_map_from_json(cfg.train_annotation_path, num_classes, cfg.classes_path)

    start_time = time.time()

    # Inference
    @tf.function
    def predict_step(img):
        """
        Runs one inference step.
        """
        return model(img, training=False)

    print("Starting evaluation...")
    for i, (image, cate_target, mask_target) in enumerate(dataset):
        # image: [1, H, W, 3]
        # cate_target: [1, K]
        # mask_target: [1, Hf, Wf, K]

        # Inference
        pred_logits, pred_masks, _ = predict_step(image)

        # Squeeze batch dim since batch_size=1
        # Process graph-optimized
        s, l, iou, gl = process_single_image_predictions(
            pred_logits[0], pred_masks[0],
            cate_target[0], mask_target[0],
            target_height=cfg.img_height,
            target_width=cfg.img_width,
            label_map=label_map,
            score_threshold=score_thresh
        )

        # Update python stats (numpy) since unique_classes needs python iteration,
        # but inputs are Tensors now.
        evaluator.update(s, l, iou, gl)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images...")

    total_time = time.time() - start_time
    print(f"Evaluation finished in {total_time:.2f}s")

    # 5. Compute mAP
    mean_ap = evaluator.calculate_map()
    print(f"mAP: {mean_ap:.4f}")

if __name__ == '__main__':
    main()