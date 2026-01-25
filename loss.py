"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains functions for the loss calculation.
"""

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment


# Graph-Compatible Hungarian Matcher

def batched_linear_sum_assignment(cost_matrix, valid_counts):
    """
    Solves the matching problem for a batch with variable number of valid targets.

    Args:
        cost_matrix (tf.Tensor): Cost matrix of shape [B, N, M].
        valid_counts (tf.Tensor): Number of valid targets per image, shape [B] int32.

    Returns:
        tuple: A tuple containing:
            - row_indices (tf.RaggedTensor): Indices into N, shape [B, (matches)].
            - col_indices (tf.RaggedTensor): Indices into M, shape [B, (matches)].
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


# Optimized Cost Calculation

def calculate_match_costs(pred_cls, gt_cls, pred_mask_logits, gt_mask,
                          lambda_cls=2.0, lambda_ce=5.0, lambda_dice=5.0):
    """
    Computes the cost matrix efficiently for Mask2Former matching.

    Args:
        pred_cls (tf.Tensor): Predicted class logits of shape [B, N, C].
        gt_cls (tf.Tensor): Ground truth one-hot labels of shape [B, M, C].
        pred_mask_logits (tf.Tensor): Predicted mask logits of shape [B, N, K] (sampled points K recommended).
        gt_mask (tf.Tensor): Ground truth masks of shape [B, M, K] (0.0 or 1.0, sampled points K).
        lambda_cls (float): Weight for classification cost. Defaults to 2.0.
        lambda_ce (float): Weight for cross-entropy cost. Defaults to 5.0.
        lambda_dice (float): Weight for Dice cost. Defaults to 5.0.

    Returns:
        tf.Tensor: Cost matrix of shape [B, N, M].
    """
    # Ensure computations happen in float32 for stability (Softplus/Exp)
    pred_cls = tf.cast(pred_cls, tf.float32)
    gt_cls = tf.cast(gt_cls, tf.float32)
    pred_mask_logits = tf.cast(pred_mask_logits, tf.float32)
    gt_mask = tf.cast(gt_mask, tf.float32)

    # Classification Cost
    # Softmax probability cost: [B, N, C] @ [B, M, C]^T -> [B, N, M]
    probs = tf.nn.softmax(pred_cls, axis=-1)
    cost_cls = -tf.linalg.matmul(probs, gt_cls, transpose_b=True)

    # Mask Cost (Sigmoid Cross Entropy)
    # Formula: (softplus(logits) - logits * targets) / K
    K = tf.cast(tf.shape(pred_mask_logits)[-1], tf.float32)

    # [B, N, K] @ [B, M, K]^T -> [B, N, M]
    interaction = tf.linalg.matmul(pred_mask_logits, gt_mask, transpose_b=True)
    pred_softplus = tf.math.softplus(pred_mask_logits)
    softplus_sum = tf.reduce_sum(pred_softplus, axis=-1)  # [B, N]
    cost_ce = (tf.expand_dims(softplus_sum, axis=2) - interaction) / K

    # Dice Cost with Laplace smoothing
    # Formula: 1 - (2*intersection + 1) / (union + 1)
    pred_mask_probs = tf.math.sigmoid(pred_mask_logits)

    # [B, N, K] @ [B, M, K]^T -> [B, N, M]
    intersection_probs = tf.linalg.matmul(pred_mask_probs, gt_mask, transpose_b=True)
    p_sum = tf.reduce_sum(pred_mask_probs, axis=-1)  # [B, N]
    g_sum = tf.reduce_sum(gt_mask, axis=-1)  # [B, M]
    denom = tf.expand_dims(p_sum, axis=2) + tf.expand_dims(g_sum, axis=1)
    cost_dice = 1.0 - (2.0 * intersection_probs + 1.0) / (denom + 1.0)

    # Weighted sum of costs
    cost = (lambda_cls * cost_cls) + (lambda_ce * cost_ce) + (lambda_dice * cost_dice)

    return cost


# Loss Functions

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """
    Computes sigmoid/softmax focal loss.

    Args:
        logits (tf.Tensor): Predicted logits.
        targets (tf.Tensor): Target labels (one-hot encoded).
        alpha (tf.Tensor, optional): Class weights. Defaults to None (uses dynamic weighting).
        gamma (float): Focusing parameter. Defaults to 2.0.

    Returns:
        tf.Tensor: Scalar focal loss value.
    """
    num_classes = tf.shape(logits)[-1]

    if alpha is None:
        # Dynamic alpha: 0.75 for BG, 0.25 for FG
        fg_weight = 0.25
        bg_weight = 0.75
        alpha_vals = tf.concat([[bg_weight], tf.fill([num_classes - 1], fg_weight)], axis=0)
        alpha = tf.cast(alpha_vals, logits.dtype)
    else:
        alpha = tf.cast(alpha, logits.dtype)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probs = tf.exp(log_probs)

    pos_weight = tf.pow(1.0 - probs, gamma)
    focal_term = -targets * alpha * pos_weight * log_probs

    return tf.reduce_sum(focal_term)


def simple_dice_loss(pred_mask, gt_mask, eps=1e-5):
    """
    Computes Dice loss on aligned (matched) pairs.

    Args:
        pred_mask (tf.Tensor): Predicted masks of shape [Total_Matches, HW].
        gt_mask (tf.Tensor): Ground truth masks of shape [Total_Matches, HW].
        eps (float): Small epsilon for numerical stability. Defaults to 1e-5.

    Returns:
        tf.Tensor: Scalar Dice loss value.
    """
    numerator = 2.0 * tf.reduce_sum(pred_mask * gt_mask, axis=1)
    denominator = tf.reduce_sum(tf.square(pred_mask), axis=1) + tf.reduce_sum(tf.square(gt_mask), axis=1)

    dice_score = (numerator + eps) / (denominator + eps)
    return tf.reduce_mean(1.0 - dice_score)


def simple_sigmoid_ce_loss(pred_logits, gt_mask):
    """
    Computes binary cross-entropy loss on aligned (matched) pairs.

    Args:
        pred_logits (tf.Tensor): Predicted logits of shape [Total_Matches, HW].
        gt_mask (tf.Tensor): Ground truth masks of shape [Total_Matches, HW].

    Returns:
        tf.Tensor: Scalar binary cross-entropy loss value.
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_mask, logits=pred_logits)
    return tf.reduce_mean(tf.reduce_mean(loss, axis=1))


def expand_targets(targets, n_indexes, m_indexes, batch_size, num_queries):
    """
    Expands matched targets to the full shape [B, N, C].

    Unmatched queries default to Background (Class 0).

    Args:
        targets (tf.Tensor): Target labels of shape [B, M, C].
        n_indexes (tf.RaggedTensor): Query indices from matching.
        m_indexes (tf.RaggedTensor): Target indices from matching.
        batch_size (int): Batch size.
        num_queries (int): Number of queries.

    Returns:
        tf.Tensor: Expanded targets of shape [B, N, C].
    """
    C = tf.shape(targets)[-1]

    # Initialize with Background (Class 0)
    bg_vec = tf.one_hot(0, C, dtype=tf.float32)
    expanded_targets = tf.broadcast_to(bg_vec, [batch_size, num_queries, C])

    # Gather matched targets
    batch_indices = n_indexes.value_rowids()
    flat_n_indices = n_indexes.values
    flat_m_indices = m_indexes.values
    gather_indices = tf.stack([batch_indices, flat_m_indices], axis=1)
    matched_targets = tf.gather_nd(targets, gather_indices)

    # Update tensor with matched targets
    scatter_indices = tf.stack([batch_indices, flat_n_indices], axis=1)
    expanded_targets = tf.tensor_scatter_nd_update(
        expanded_targets,
        scatter_indices,
        matched_targets
    )
    return expanded_targets


# Main Loss Orchestrator

def mask2former_loss(
        cls_pred,           # [B, N, num_classes]
        mask_pred_logits,   # [B, H, W, N]
        cls_target,         # [B, M, num_classes] (One-hot)
        gt_masks,           # [B, H, W, M]
        valid_counts,       # [B]
        gt_is_padding,      # [B, M]
        cls_loss_weight=2.0,
        mask_loss_weight=5.0,
        dice_loss_weight=5.0
):
    """
    Calculates Mask2Former loss efficiently.

    Args:
        cls_pred (tf.Tensor): Predicted class logits of shape [B, N, num_classes].
        mask_pred_logits (tf.Tensor): Predicted mask logits of shape [B, H, W, N].
        cls_target (tf.Tensor): Ground truth class labels (one-hot) of shape [B, M, num_classes].
        gt_masks (tf.Tensor): Ground truth masks of shape [B, H, W, M].
        valid_counts (tf.Tensor): Number of valid targets per image, shape [B].
        gt_is_padding (tf.Tensor): Boolean mask indicating padding, shape [B, M].
        cls_loss_weight (float, optional): Weight for classification loss. Defaults to 2.0.
        mask_loss_weight (float, optional): Weight for mask cross-entropy loss. Defaults to 1.0.
        dice_loss_weight (float, optional): Weight for Dice loss. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - total_loss (tf.Tensor): Weighted sum of all losses.
            - loss_cls (tf.Tensor): Classification loss.
            - loss_dice_val (tf.Tensor): Dice loss.
            - loss_ce_val (tf.Tensor): Cross-entropy loss.
    """
    # Pre-processing and sorting
    # Sort targets so valid objects are first
    sorted_indices = tf.argsort(tf.cast(gt_is_padding, tf.int32), axis=1, direction='ASCENDING')
    cls_target_sorted = tf.gather(cls_target, sorted_indices, batch_dims=1, axis=1)

    # Flatten GT masks: [B, H, W, M] -> [B, M, HW]
    B = tf.shape(mask_pred_logits)[0]
    H = tf.shape(mask_pred_logits)[1]
    W = tf.shape(mask_pred_logits)[2]
    N = tf.shape(mask_pred_logits)[3]

    gt_masks_t = tf.transpose(gt_masks, perm=[0, 3, 1, 2])
    gt_masks_sorted_t = tf.gather(gt_masks_t, sorted_indices, batch_dims=1, axis=1)
    gt_masks_flat = tf.reshape(gt_masks_sorted_t, [B, -1, H * W])
    gt_masks_flat = tf.cast(gt_masks_flat, tf.float32)

    # Flatten pred masks: [B, H, W, N] -> [B, N, HW]
    mask_pred_logits_t = tf.transpose(mask_pred_logits, perm=[0, 3, 1, 2])
    mask_pred_flat_logits = tf.reshape(mask_pred_logits_t, [B, N, H*W])
    mask_pred_flat_probs = tf.nn.sigmoid(mask_pred_flat_logits)

    # Matching
    cost_matrix = calculate_match_costs(
        cls_pred, cls_target_sorted, mask_pred_flat_logits, gt_masks_flat
    )
    row_indices, col_indices = batched_linear_sum_assignment(cost_matrix, valid_counts)

    # Classification loss for all queries
    expanded_targets = expand_targets(cls_target_sorted, row_indices, col_indices, B, N)
    loss_cls = focal_loss(cls_pred, expanded_targets, alpha=0.5)

    # Mask losses for matched pairs only
    batch_ids = row_indices.value_rowids()
    flat_n = row_indices.values
    flat_m = col_indices.values

    # Global indices for flattened batch
    global_n_idx = tf.cast(batch_ids, tf.int32) * N + tf.cast(flat_n, tf.int32)
    global_m_idx = tf.cast(batch_ids, tf.int32) * tf.shape(gt_masks_flat)[1] + tf.cast(flat_m, tf.int32)

    # Flatten and gather matched pairs
    flat_pred_logits = tf.reshape(mask_pred_flat_logits, [-1, H*W])
    flat_pred_probs  = tf.reshape(mask_pred_flat_probs, [-1, H*W])
    flat_gt_masks    = tf.reshape(gt_masks_flat, [-1, H*W])

    matched_pred_logits = tf.gather(flat_pred_logits, global_n_idx)
    matched_pred_probs  = tf.gather(flat_pred_probs, global_n_idx)
    matched_gt_masks    = tf.gather(flat_gt_masks, global_m_idx)

    # Calculate losses on matched pairs
    num_matches = tf.maximum(tf.cast(tf.shape(matched_gt_masks)[0], tf.float32), 1.0)

    loss_dice_val = simple_dice_loss(matched_pred_probs, matched_gt_masks)
    loss_ce_val = simple_sigmoid_ce_loss(matched_pred_logits, matched_gt_masks)
    loss_cls = loss_cls / num_matches

    total_loss = (cls_loss_weight * loss_cls +
                  dice_loss_weight * loss_dice_val +
                  mask_loss_weight * loss_ce_val)

    return total_loss, loss_cls, loss_dice_val, loss_ce_val


# Entry Point

def compute_multiscale_loss(pred_logits, pred_masks, class_target, mask_target, num_classes, aux_outputs=None):
    """
    Main entry point for computing multi-scale Mask2Former loss.

    Args:
        pred_logits (tf.Tensor): Final predicted class logits.
        pred_masks (tf.Tensor): Final predicted masks.
        class_target (tf.Tensor): Ground truth class indices of shape [B, M]. Padding is -1.
        mask_target (tf.Tensor): Ground truth masks of shape [B, H, W, M].
        num_classes (int): Number of object classes (excluding background).
        aux_outputs (list, optional): List of auxiliary outputs from intermediate decoder layers. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - total_loss (tf.Tensor): Average total loss across all layers.
            - total_cate_loss (tf.Tensor): Average classification loss across all layers.
            - total_dice_loss (tf.Tensor): Average Dice loss across all layers.
            - total_mask_loss (tf.Tensor): Average mask cross-entropy loss across all layers.
    """
    # Handle invalid class indices
    invalid_cls_mask = class_target >= num_classes
    class_target_safe = tf.where(invalid_cls_mask, -1, class_target)

    # Mask invalid targets
    invalid_masks_mask = tf.expand_dims(tf.expand_dims(invalid_cls_mask, 1), 1)
    invalid_masks_mask = tf.broadcast_to(invalid_masks_mask, tf.shape(mask_target))
    mask_target_safe = tf.where(invalid_masks_mask, tf.zeros_like(mask_target), mask_target)

    # Determine padding and valid counts
    is_padding = tf.equal(class_target_safe, -1)
    valid_counts = tf.reduce_sum(tf.cast(tf.math.logical_not(is_padding), tf.int32), axis=1)

    # One-hot targets: map classes to 1..C, padding (-1) maps to 0
    class_true_one_hot = tf.one_hot(class_target_safe + 1, depth=num_classes + 1)

    # Collect outputs from all decoder layers
    pred_list_cls = []
    pred_list_masks = []
    if aux_outputs is not None:
        for layer_output in aux_outputs:
            pred_list_cls.append(layer_output["pred_logits"])
            pred_list_masks.append(layer_output["pred_masks"])
    pred_list_cls.append(pred_logits)
    pred_list_masks.append(pred_masks)

    # Accumulate loss across layers
    total_loss = 0.0
    total_cate_loss = 0.0
    total_dice_loss = 0.0
    total_mask_loss = 0.0

    num_layers = tf.cast(len(pred_list_cls), tf.float32)

    for cls_pred_i, mask_pred_i in zip(pred_list_cls, pred_list_masks):
        # Ensure mask shape is [B, H, W, N]
        if mask_pred_i.shape[1] == tf.shape(cls_pred_i)[1]:
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