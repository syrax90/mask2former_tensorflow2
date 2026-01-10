"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains functions for the loss calculation.
"""

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# Helper: Graph-Compatible Hungarian Matcher
# ==============================================================================

def batched_linear_sum_assignment(cost_matrix, valid_counts):
    """
    Solves the matching problem for a batch with variable number of valid targets.
    Args:
        cost_matrix: [B, N, M]
        valid_counts: [B] int32, number of valid targets per image.
    Returns:
        row_indices: RaggedTensor [B, (matches)] (Indices into N)
        col_indices: RaggedTensor [B, (matches)] (Indices into M)
    """
    def solve_hungarian(cost, counts):
        b = cost.shape[0]

        row_list = []
        col_list = []
        row_lengths = []

        for i in range(b):
            cnt = counts[i]
            if cnt > 0:
                # Slice only valid targets [N, cnt]
                c_valid = cost[i, :, :cnt]
                r, c = linear_sum_assignment(c_valid)
                row_list.append(r.astype(np.int32))
                col_list.append(c.astype(np.int32))
                row_lengths.append(len(r))
            else:
                row_lengths.append(0)

        # Concatenate for RaggedTensor reconstruction
        flat_rows = np.concatenate(row_list) if row_list else np.array([], dtype=np.int32)
        flat_cols = np.concatenate(col_list) if col_list else np.array([], dtype=np.int32)

        return flat_rows, flat_cols, np.array(row_lengths, dtype=np.int32)

    # Execute inside py_function
    flat_rows, flat_cols, row_lengths = tf.py_function(
        func=solve_hungarian,
        inp=[cost_matrix, valid_counts],
        Tout=[tf.int32, tf.int32, tf.int32]
    )

    # Reconstruct RaggedTensors
    row_indices = tf.RaggedTensor.from_row_lengths(flat_rows, row_lengths)
    col_indices = tf.RaggedTensor.from_row_lengths(flat_cols, row_lengths)

    return row_indices, col_indices


# ==============================================================================
# 1. Optimized Cost Calculation
# ==============================================================================

def calculate_match_costs(pred_cls, gt_cls, pred_mask_logits, gt_mask,
                          lambda_cls=2.0, lambda_ce=5.0, lambda_dice=5.0):
    """
    Computes the cost matrix efficiently for Mask2Former.

    Args:
        pred_cls: [B, N, C] - Predicted class logits (or probs if pre-softmaxed, check logic)
        gt_cls: [B, M, C] - Ground Truth One-Hot labels
        pred_mask_logits: [B, N, K] - Predicted mask logits (Sampled points K recommended)
        gt_mask: [B, M, K] - Ground truth masks (0.0 or 1.0, Sampled points K)

    Returns:
        cost_matrix: [B, N, M]
    """
    # Ensure computations happen in float32 for stability (Softplus/Exp)
    pred_cls = tf.cast(pred_cls, tf.float32)
    gt_cls = tf.cast(gt_cls, tf.float32)
    pred_mask_logits = tf.cast(pred_mask_logits, tf.float32)
    gt_mask = tf.cast(gt_mask, tf.float32)

    # --- 1. Classification Cost ---
    # Mask2Former Matcher uses Softmax probability for the cost (not Focal Loss)
    # pred_cls usually contains logits.
    probs = tf.nn.softmax(pred_cls, axis=-1)

    # Cost = -prob[target_class]
    # optimize: [B, N, C] @ [B, M, C]^T -> [B, N, M]
    cost_cls = -tf.linalg.matmul(probs, gt_cls, transpose_b=True)

    # --- 2. Mask Cost (Sigmoid Cross Entropy) ---
    # We want: sigmoid_cross_entropy(logits, targets)
    # Loss formula: softplus(logits) - logits * targets
    # Summed over sample points (K), divided by K.

    # K = Number of sampled points (or HW)
    K = tf.cast(tf.shape(pred_mask_logits)[-1], tf.float32)

    # Interaction term: sum(logits * gt) over K
    # optimize: [B, N, K] @ [B, M, K]^T -> [B, N, M]
    interaction = tf.linalg.matmul(pred_mask_logits, gt_mask, transpose_b=True)

    # Pred term: sum(softplus(logits)) over K
    pred_softplus = tf.math.softplus(pred_mask_logits)
    softplus_sum = tf.reduce_sum(pred_softplus, axis=-1)  # [B, N]

    # Combine: (sum(softplus) - sum(logits*gt)) / K
    # shape: ([B, N, 1] - [B, N, M])
    cost_ce = (tf.expand_dims(softplus_sum, axis=2) - interaction) / K

    # --- 3. Dice Cost ---
    # Mask2Former uses: 1 - (2*intersection + 1) / (union + 1) (Laplace smoothing)

    pred_mask_probs = tf.math.sigmoid(pred_mask_logits)

    # Intersection: sum(p * g)
    # optimize: [B, N, K] @ [B, M, K]^T -> [B, N, M]
    intersection_probs = tf.linalg.matmul(pred_mask_probs, gt_mask, transpose_b=True)

    # Denominator: sum(p) + sum(g)
    p_sum = tf.reduce_sum(pred_mask_probs, axis=-1)  # [B, N]
    g_sum = tf.reduce_sum(gt_mask, axis=-1)  # [B, M]

    # denom: [B, N, 1] + [B, 1, M] -> [B, N, M]
    denom = tf.expand_dims(p_sum, axis=2) + tf.expand_dims(g_sum, axis=1)

    # Smoothed Dice Cost
    cost_dice = 1.0 - (2.0 * intersection_probs + 1.0) / (denom + 1.0)

    # --- Weighted Sum ---
    cost = (lambda_cls * cost_cls) + (lambda_ce * cost_ce) + (lambda_dice * cost_dice)

    return cost


# ==============================================================================
# 2. Loss Functions (Simplified)
# ==============================================================================

def focal_loss(logits, targets, alpha=None, gamma=2.0, eps=1e-7):
    """
    Sigmoid/Softmax Focal Loss.
    """
    num_classes = tf.shape(logits)[-1]

    if alpha is None:
        # Dynamic alpha: 0.1 for BG (index 0), 1.0 for FG
        fg_weight = 0.25
        bg_weight = 0.75
        alpha_vals = tf.concat([[bg_weight], tf.fill([num_classes - 1], fg_weight)], axis=0)
        alpha = tf.cast(alpha_vals, logits.dtype)
    else:
        alpha = tf.cast(alpha, logits.dtype)

    # Use log_softmax for better numerical stability
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probs = tf.exp(log_probs)

    # Focal term: -alpha * (1 - p)^gamma * log(p) * target
    pos_weight = tf.pow(1.0 - probs, gamma)
    focal_term = -targets * alpha * pos_weight * log_probs

    #return tf.reduce_mean(tf.reduce_sum(focal_term, axis=-1))
    return tf.reduce_sum(focal_term)


def simple_dice_loss(pred_mask, gt_mask, eps=1e-5):
    """
    Computes Dice loss on aligned (matched) pairs.
    pred_mask, gt_mask: [Total_Matches, HW]
    """
    numerator = 2.0 * tf.reduce_sum(pred_mask * gt_mask, axis=1)
    denominator = tf.reduce_sum(tf.square(pred_mask), axis=1) + tf.reduce_sum(tf.square(gt_mask), axis=1)

    dice_score = (numerator + eps) / (denominator + eps)
    return tf.reduce_mean(1.0 - dice_score)


def simple_sigmoid_ce_loss(pred_logits, gt_mask):
    """
    Computes Binary CE loss on aligned (matched) pairs.
    pred_logits, gt_mask: [Total_Matches, HW]
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_mask, logits=pred_logits)
    # Mean over spatial dims, then mean over batch
    return tf.reduce_mean(tf.reduce_mean(loss, axis=1))


def expand_targets(targets, n_indexes, m_indexes, batch_size, num_queries):
    """
    Expands matched targets to the full shape [B, N, C].
    Unmatched queries default to Background (Class 0).
    """
    C = tf.shape(targets)[-1]

    # 1. Initialize with Background (Class 0)
    bg_vec = tf.one_hot(0, C, dtype=tf.float32)
    expanded_targets = tf.broadcast_to(bg_vec, [batch_size, num_queries, C])

    # 2. Gather indices
    batch_indices = n_indexes.value_rowids()
    flat_n_indices = n_indexes.values
    flat_m_indices = m_indexes.values

    # 3. Gather matched targets
    gather_indices = tf.stack([batch_indices, flat_m_indices], axis=1)
    matched_targets = tf.gather_nd(targets, gather_indices)

    # 4. Update tensor
    scatter_indices = tf.stack([batch_indices, flat_n_indices], axis=1)

    # Optimization: tensor_scatter_nd_update is efficient here
    expanded_targets = tf.tensor_scatter_nd_update(
        expanded_targets,
        scatter_indices,
        matched_targets
    )
    return expanded_targets


# ==============================================================================
# 3. Main Loss Orchestrator
# ==============================================================================

def mask2former_loss(
        cls_pred,           # [B, N, num_classes]
        mask_pred_logits,   # [B, H, W, N]
        cls_target,         # [B, M, num_classes] (One-hot)
        gt_masks,           # [B, H, W, M]
        valid_counts,       # [B]
        gt_is_padding,      # [B, M]
        cls_loss_weight=2.0,
        mask_loss_weight=1.0,
        dice_loss_weight=1.0
):
    """
    Calculates Mask2Former loss efficiently.
    """
    # --- 1. Pre-processing & Sorting ---
    # Sort targets so valid objects are first (required for valid_counts logic)
    sorted_indices = tf.argsort(tf.cast(gt_is_padding, tf.int32), axis=1, direction='ASCENDING')
    cls_target_sorted = tf.gather(cls_target, sorted_indices, batch_dims=1, axis=1)

    # Sort and Flatten GT Masks: [B, H, W, M] -> [B, HW, M] -> Transpose [B, M, HW]
    B = tf.shape(mask_pred_logits)[0]
    H = tf.shape(mask_pred_logits)[1]
    W = tf.shape(mask_pred_logits)[2]
    N = tf.shape(mask_pred_logits)[3]

    # Transpose GT to [B, M, H, W] for easier gathering then flatten
    gt_masks_t = tf.transpose(gt_masks, perm=[0, 3, 1, 2])
    gt_masks_sorted_t = tf.gather(gt_masks_t, sorted_indices, batch_dims=1, axis=1)
    gt_masks_flat = tf.reshape(gt_masks_sorted_t, [B, -1, H * W]) # [B, M, HW]
    gt_masks_flat = tf.cast(gt_masks_flat, tf.float32)

    # Flatten Pred Masks: [B, H, W, N] -> [B, HW, N] -> Transpose [B, N, HW]
    mask_pred_logits_t = tf.transpose(mask_pred_logits, perm=[0, 3, 1, 2]) # [B, N, H, W]
    mask_pred_flat_logits = tf.reshape(mask_pred_logits_t, [B, N, H*W])
    mask_pred_flat_probs = tf.nn.sigmoid(mask_pred_flat_logits)

    # --- 2. Matching ---
    cost_matrix = calculate_match_costs(
        cls_pred, cls_target_sorted, mask_pred_flat_logits, gt_masks_flat
    )

    # Returns Ragged Indices
    row_indices, col_indices = batched_linear_sum_assignment(cost_matrix, valid_counts)

    # --- 3. Classification Loss (All Queries) ---
    # Construct full target tensor [B, N, C]
    expanded_targets = expand_targets(cls_target_sorted, row_indices, col_indices, B, N)
    loss_cls = focal_loss(cls_pred, expanded_targets)

    # --- 4. Mask Losses (Matched Pairs Only) ---
    # Gather pairs efficiently without recalculating indices multiple times

    # Convert Ragged indices to global flat indices for gather
    batch_ids = row_indices.value_rowids()
    flat_n = row_indices.values
    flat_m = col_indices.values

    # Global indices for flattened batch [B * N] and [B * M]
    global_n_idx = tf.cast(batch_ids, tf.int32) * N + tf.cast(flat_n, tf.int32)
    global_m_idx = tf.cast(batch_ids, tf.int32) * tf.shape(gt_masks_flat)[1] + tf.cast(flat_m, tf.int32)

    # Flatten batch dim for gathering
    # Preds: [B, N, HW] -> [B*N, HW]
    flat_pred_logits = tf.reshape(mask_pred_flat_logits, [-1, H*W])
    flat_pred_probs  = tf.reshape(mask_pred_flat_probs, [-1, H*W])
    # GT: [B, M, HW] -> [B*M, HW]
    flat_gt_masks    = tf.reshape(gt_masks_flat, [-1, H*W])

    # Gather
    matched_pred_logits = tf.gather(flat_pred_logits, global_n_idx)
    matched_pred_probs  = tf.gather(flat_pred_probs, global_n_idx)
    matched_gt_masks    = tf.gather(flat_gt_masks, global_m_idx)

    # Calculate Losses on matched pairs
    # Note: Matches are guaranteed to be valid objects due to valid_counts in matching
    num_matches = tf.maximum(tf.cast(tf.shape(matched_gt_masks)[0], tf.float32), 1.0)

    # Dice
    loss_dice_val = simple_dice_loss(matched_pred_probs, matched_gt_masks)

    # CE / Sigmoid
    loss_ce_val = simple_sigmoid_ce_loss(matched_pred_logits, matched_gt_masks)

    # Normalize by num_matches?
    # Usually Dice is mean over pairs, CE is mean over pixels then pairs.
    # The simple_ implementations return mean over pairs.
    loss_cls = loss_cls / num_matches

    total_loss = (cls_loss_weight * loss_cls +
                  dice_loss_weight * loss_dice_val +
                  mask_loss_weight * loss_ce_val)

    return total_loss, loss_cls, loss_dice_val, loss_ce_val


# ==============================================================================
# 4. Entry Point
# ==============================================================================

def compute_multiscale_loss(pred_logits, pred_masks, class_target, mask_target, num_classes, aux_outputs=None):
    """
    Main entry point.
    class_target: [B, M] integers. Padding is -1.
    mask_target: [B, H, W, M]
    """
    # 1. Handle invalid class indices
    invalid_cls_mask = class_target >= num_classes
    class_target_safe = tf.where(invalid_cls_mask, -1, class_target)

    # Mask invalid mask targets (optional, but good for safety)
    # We rely on is_padding for logic, but zeroing out helps debug/safety
    invalid_masks_mask = tf.expand_dims(tf.expand_dims(invalid_cls_mask, 1), 1)
    invalid_masks_mask = tf.broadcast_to(invalid_masks_mask, tf.shape(mask_target))
    mask_target_safe = tf.where(invalid_masks_mask, tf.zeros_like(mask_target), mask_target)

    # 2. Determine Padding
    is_padding = tf.equal(class_target_safe, -1)
    valid_counts = tf.reduce_sum(tf.cast(tf.math.logical_not(is_padding), tf.int32), axis=1)

    # 3. One-hot targets (inc. background/padding handling for now)
    # Map classes to 1..C. Padding (-1) maps to 0.
    class_true_one_hot = tf.one_hot(class_target_safe + 1, depth=num_classes + 1)

    # 4. Collect Outputs
    pred_list_cls = []
    pred_list_masks = []
    if aux_outputs is not None:
        for layer_output in aux_outputs:
            pred_list_cls.append(layer_output["pred_logits"])
            pred_list_masks.append(layer_output["pred_masks"])
    pred_list_cls.append(pred_logits)
    pred_list_masks.append(pred_masks)

    # 5. Accumulate Loss
    total_loss = 0.0
    total_cate_loss = 0.0
    total_dice_loss = 0.0
    total_mask_loss = 0.0

    num_layers = tf.cast(len(pred_list_cls), tf.float32)

    for cls_pred_i, mask_pred_i in zip(pred_list_cls, pred_list_masks):
        # Ensure mask shape is [B, H, W, N]
        if mask_pred_i.shape[1] == tf.shape(cls_pred_i)[1]:
             # If shape is [B, N, H, W] -> transpose to [B, H, W, N]
            mask_pred_i = tf.transpose(mask_pred_i, perm=[0, 2, 3, 1])

        l_total, l_cls, l_dice, l_ce = mask2former_loss(
            cls_pred_i,
            mask_pred_i,
            class_true_one_hot,
            mask_target_safe,
            valid_counts,
            is_padding
        )

        total_loss += l_total
        total_cate_loss += l_cls
        total_dice_loss += l_dice
        total_mask_loss += l_ce

    return (total_loss / num_layers,
            total_cate_loss / num_layers,
            total_dice_loss / num_layers,
            total_mask_loss / num_layers)