"""
Author: Pavel Timonin
Created: 2026-01-17
Description: This script calculates Mean Average Precision (mAP) for the Mask2Former model on the validation dataset.
"""

import os
import time
import tensorflow as tf
from config import Mask2FormerConfig
from coco_dataset_optimized import create_coco_tfrecord_dataset, get_classes
from model_functions import Mask2FormerModel

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
    score_threshold=0.0
):
    """
    Processes predictions for a single image to compute IoUs and matches.
    Run in Graph Mode.

    Args:
        pred_logits: [Q, num_classes+1]
        pred_masks:  [Q, H_feat, W_feat]
        gt_cate:     [sum(S_i^2)] (flat list of categories for grid cells) - NOT USED directly here
                     Wait, the input from dataset is just `cate_target` which is per-grid-cell.
                     We need per-instance GT for evaluation.

                     The metric evaluation usually compares *predicted instances* vs *ground truth instances*.
                     The dataset pipeline `create_coco_tfrecord_dataset` outputs *training targets*
                     (dense grid targets), NOT the raw sparse ground truth instances needed for evaluation.

                     However, the task request says:
                     "Use create_coco_tfrecord_dataset without shuffling and augmentation."

                     So we must reconstruct "GT instances" from the grid targets or accept that
                     we are evaluating against the grid-assigned targets.

                     Let's look at `create_coco_tfrecord_dataset`. It returns:
                        - cate_targets: [B, sum(S_i^2)]
                        - mask_targets: [B, Hf, Wf, sum(S_i^2)]

                     The `mask_targets` channel dimension IS the number of grid assignments.
                     `cate_targets` holds the class ID for each channel.
                     Typically many are -1 (no object). We should filter those out to get the "GT objects".

        target_height: int
        target_width: int
        score_threshold: float

    Returns:
        pred_scores: [N_kept]
        pred_classes: [N_kept]
        iou_matrix: [N_kept, N_gt_instances]
        gt_classes: [N_gt_instances]
    """

    # --- 1. Process Predictions ---
    # Softmax on classes
    scores = tf.nn.softmax(pred_logits, axis=-1)[:, :-1] # [Q, num_classes] (exclude background)

    # Get max score and class per query
    max_scores = tf.reduce_max(scores, axis=-1)
    pred_labels = tf.argmax(scores, axis=-1)

    # Filter by score threshold (optional, or keep top K)
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
    # gt_cate: [K_grid]
    # gt_mask: [Hf, Wf, K_grid] -> Transpose to [K_grid, Hf, Wf] first?
    # Wait, dataset outputs mask as [B, Hf, Wf, K]. One image: [Hf, Wf, K].

    # Filter out empty targets (-1)
    valid_gt_indices = tf.where(gt_cate_dataset > -1)[:, 0]

    gt_labels = tf.gather(gt_cate_dataset, valid_gt_indices)
    gt_masks_grid = tf.gather(gt_mask_dataset, valid_gt_indices, axis=-1) # [Hf, Wf, N_gt]

    # The GT masks from dataset are already smaller scale (output of parse_example resizes them).
    # But wait, `create_coco_tfrecord_dataset` resizes masks to `target_mask_height` which is `target_height * scale`.
    # Prediction masks (from model) are typically low res (1/4 stride usually).
    # To compute IoU accurately, we should bring both to the same size.
    # The dataset `mask_targets` are resized to `scale` (e.g. 0.25).
    # If `scale` in config is 0.25, then they are 1/4 size.
    # The model output `pred_masks` is also low res.
    # Matching them at low res is faster and sufficient for training, but for mAP we usually want input resolution.
    # Let's upsample BOTH to `target_height, target_width` to be safe and standard.

    gt_masks_grid_exp = tf.expand_dims(gt_masks_grid, 0) # [1, Hf, Wf, N_gt]
    gt_masks_up = tf.image.resize(gt_masks_grid_exp, (target_height, target_width), method='nearest')
    gt_masks_up = tf.squeeze(gt_masks_up, 0) # [H, W, N_gt]
    gt_masks_up = tf.transpose(gt_masks_up, perm=[2, 0, 1]) # [N_gt, H, W]

    gt_masks_bin = tf.cast(tf.cast(gt_masks_up, tf.float32) > 0.5, tf.float32)

    # --- 3. Compute IoU ---
    # If no predictions or no GT, return empty

    n_pred = tf.shape(kept_scores)[0]
    n_gt = tf.shape(gt_labels)[0]

    if n_pred == 0 or n_gt == 0:
        return kept_scores, kept_labels, tf.zeros((n_pred, n_gt)), gt_labels

    iou_mat = compute_iou(pred_masks_bin, gt_masks_bin) # [N_pred, N_gt]

    return kept_scores, kept_labels, iou_mat, gt_labels


@tf.function
def compute_matches_tf(iou_matrix, iou_thresh):
    """
    Greedy matching of predictions to ground truth using TensorFlow.

    Args:
        iou_matrix: [N_dt, N_gt] float32. Rows are predictions (sorted), cols are GT.
        iou_thresh: float scalar.

    Returns:
        dt_matched: [N_dt] bool.
    """
    n_dt = tf.shape(iou_matrix)[0]
    n_gt = tf.shape(iou_matrix)[1]

    gt_covered = tf.zeros(n_gt, dtype=tf.bool)
    dt_matched = tf.zeros(n_dt, dtype=tf.bool)

    if n_gt == 0 or n_dt == 0:
        return dt_matched

    def body(i, gt_cov, dt_mat):
        # i: current dt index (already sorted by score)
        row = iou_matrix[i] # [N_gt]

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

    _, _, dt_matched_final = tf.while_loop(
        lambda i, g, d: i < n_dt,
        body,
        [0, gt_covered, dt_matched],
        parallel_iterations=1 # Serial execution required for greedy matching
    )

    return dt_matched_final

class MAPEvaluator:
    """
    TensorFlow-optimized mAP Evaluator.
    Accumulates results in lists of Tensors, computes mAP using TF ops.
    """
    def __init__(self, num_classes, iou_thresholds=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else tf.linspace(0.5, 0.95, 10)
        # Convert thresholds to tensor explicitly if not already
        if not tf.is_tensor(self.iou_thresholds): # pragma: no cover
             self.iou_thresholds = tf.convert_to_tensor(self.iou_thresholds, dtype=tf.float32)

        # Storage: { cls_id: { 'scores': [list_of_tensors], 'tp': {thresh: [list_of_tensors]}, 'n_gt': int } }
        # n_gt is just an integer counter, no need for tensor overhead there
        self.stats = {}
        for c in range(num_classes):
            self.stats[c] = {
                'scores': [],
                'tp': {t_idx: [] for t_idx in range(len(self.iou_thresholds))},
                'n_gt': 0
            }

    @tf.function(reduce_retracing=True)
    def _compute_update_batch(self, pred_scores, pred_labels, iou_matrix, gt_labels):
        """
        Computes matches for a single batch (image) using TF graph.
        Returns updates required for stats.
        """
        # Group by class
        # Ideally we loop over classes present in the batch
        # But for graph mode efficienty over fixed classes, we might need a fixed loop?
        # Actually, dynamic loop over unique classes is better.

        unique_classes, _ = tf.unique(tf.concat([pred_labels, gt_labels], axis=0))

        # We can't return arbitrary structure easily.
        # So we process everything here and return packed tensors?
        # Or just return the raw needed data?
        # Actually, `compute_matches_tf` needs to be called per class.

        return unique_classes

    def update(self, pred_scores, pred_labels, iou_matrix, gt_labels):
        """
        Update stats. Inputs are Tensors.
        """
        # Logic partially in Python to manage the Class dictionary structure,
        # but heavy lifting in TF.

        # 1. Get unique classes in this image
        # Using TF for uniqueness
        all_labels = tf.concat([tf.cast(pred_labels, tf.int32), gt_labels], axis=0)
        unique_classes, _ = tf.unique(all_labels)
        unique_classes = unique_classes.numpy() # iterate in python

        for cls_id in unique_classes:
            cls_id = int(cls_id)
            if cls_id not in self.stats:
                continue

            # tf.gather indices
            gt_indices = tf.where(tf.equal(gt_labels, cls_id))[:, 0]
            dt_indices = tf.where(tf.equal(pred_labels, cls_id))[:, 0]

            n_gt = tf.shape(gt_indices)[0].numpy()
            self.stats[cls_id]['n_gt'] += n_gt

            if tf.shape(dt_indices)[0] == 0:
                continue

            # Gather data for this class
            # IoU matrix sub-block: [N_dt_cls, N_gt_cls]
            cls_iou_mat = tf.gather(iou_matrix, dt_indices) # rows
            if n_gt > 0:
                cls_iou_mat = tf.gather(cls_iou_mat, gt_indices, axis=1) # cols
            else:
                cls_iou_mat = tf.zeros((tf.shape(dt_indices)[0], 0), dtype=tf.float32)

            cls_scores = tf.gather(pred_scores, dt_indices)

            # Sort by score desc
            sort_idx = tf.argsort(cls_scores, direction='DESCENDING')
            cls_scores_sorted = tf.gather(cls_scores, sort_idx)
            cls_iou_mat_sorted = tf.gather(cls_iou_mat, sort_idx)

            # Store scores
            self.stats[cls_id]['scores'].append(cls_scores_sorted)

            # Check matches for each threshold
            for t_idx, t_val in enumerate(self.iou_thresholds):
                if n_gt == 0:
                    matches = tf.zeros(tf.shape(dt_indices)[0], dtype=tf.bool)
                else:
                    matches = compute_matches_tf(cls_iou_mat_sorted, t_val)

                self.stats[cls_id]['tp'][t_idx].append(matches)

    @tf.function(reduce_retracing=True)
    def _compute_ap_per_class(self, tp_cat, n_gt):
        # tp_cat: [N_total_dt] bool
        # n_gt: int scalar (tensor)

        n_gt = tf.cast(n_gt, tf.float32)

        # In this sorted list (globally sorted), compute cumsum
        tp = tf.cast(tp_cat, tf.float32)
        fp = 1.0 - tp

        tp_sum = tf.cumsum(tp)
        fp_sum = tf.cumsum(fp)

        recalls = tp_sum / n_gt
        precisions = tp_sum / (tp_sum + fp_sum)

        # Monotonic decreasing methods
        # Reverse, accumulate max, reverse
        precisions = tf.reverse(tf.math.cumprod(tf.reverse(precisions, axis=[0]), exclusive=False, reverse=False), axis=[0])
        # Wait, cumprod is wrong. accumulated max!
        # TF doesnt have naive accumulate_max.
        # hack: scan?
        precisions = tf.scan(lambda a, x: tf.maximum(a, x), precisions, reverse=True)

        # 101-point AP
        recall_steps = tf.linspace(0.0, 1.0, 101)

        # For each step, find max precision with recall >= step
        # Since precisions is monotonic (processed above) and recalls is increasing:
        # We can find index where recall >= step

        # searchsorted works on sorted sequences. recalls is strictly increasing (mostly).
        # indices = tf.searchsorted(recalls, recall_steps, side='left')

        # BUT: recalls might not be strictly increasing if tp doesn't change?
        # No, tp_sum is non-decreasing. recall is non-decreasing.

        indices = tf.searchsorted(recalls, recall_steps, side='left')

        # Gather precisions
        # If index == len, output 0
        N = tf.shape(precisions)[0]

        # Handle out of bounds
        indices_clipped = tf.minimum(indices, N - 1)
        gathered = tf.gather(precisions, indices_clipped)

        # If index was N (not found), value should be 0.
        mask = indices < N
        gathered = tf.where(mask, gathered, 0.0)

        return tf.reduce_mean(gathered)

    def calculate_map(self):
        aps = []
        print("Computing mAP...")
        for cls_id in self.stats:
            n_gt = self.stats[cls_id]['n_gt']
            if n_gt == 0:
                continue

            # Global sort for this class
            # We have list of score tensors. Concat them.
            scores_list = self.stats[cls_id]['scores']
            if not scores_list:
                aps.append(0.0)
                continue

            all_scores = tf.concat(scores_list, axis=0) # [Total_DT]

            # Sort globally
            global_sort_idx = tf.argsort(all_scores, direction='DESCENDING')

            # Compute AP per threshold
            ap_thresholds = []
            for t_idx in self.stats[cls_id]['tp']:
                tp_list = self.stats[cls_id]['tp'][t_idx]
                all_matches = tf.concat(tp_list, axis=0) # [Total_DT]

                # Reorder using global sort
                all_matches_sorted = tf.gather(all_matches, global_sort_idx)

                ap = self._compute_ap_per_class(all_matches_sorted, tf.constant(n_gt))
                ap_thresholds.append(ap)

            # Mean for this class over thresholds
            cls_mAP = tf.reduce_mean(tf.stack(ap_thresholds))
            aps.append(cls_mAP)

        if not aps:
            return 0.0

        return tf.reduce_mean(tf.stack(aps)).numpy()



def main():
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
    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)

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

    start_time = time.time()

    # Inference
    @tf.function
    def predict_step(img):
        return model(img, training=False)

    print("Starting evaluation...")
    for i, (image, cate_target, mask_target) in enumerate(dataset):
        # image: [1, H, W, 3]
        # cate_target: [1, K]
        # mask_target: [1, Hf, Wf, K]

        # Inference
        pred_logits, pred_masks, _ = predict_step(image)

        # Squeeze batch dim since batch_size=1
        img_single = image[0]
        # Process graph-optimized
        s, l, iou, gl = process_single_image_predictions(
            pred_logits[0], pred_masks[0],
            cate_target[0], mask_target[0],
            target_height=cfg.img_height,
            target_width=cfg.img_width,
            num_classes=num_classes,
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
    mAP = evaluator.calculate_map()
    print(f"mAP: {mAP:.4f}")

if __name__ == '__main__':
    main()
