"""
Author: Pavel Timonin
Created: 2026-01-17
Description: This script calculates Mean Average Precision (mAP) for the Mask2Former model on the validation dataset.
"""

import time
import json
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from config import Mask2FormerConfig
from model_functions import Mask2FormerModel
from coco_dataset_optimized import COCOAnalysis, create_coco_eval_dataset
from reassign_categories import reassign_category_ids


def main():
    """
    Main function to run standard mAP evaluation.

    Loads the validation dataset and model, runs inference, and computes mAP
    using pycocotools.
    """
    cfg = Mask2FormerConfig()

    # Setup Data
    print(f"Loading test dataset from: {cfg.tfrecord_test_path}")
    dataset = create_coco_eval_dataset(
        cfg.tfrecord_test_path,
        target_size=(cfg.img_height, cfg.img_width),
        batch_size=1,  # Process one by one to handle resizing back to original dims easily
        scale=cfg.image_scales[0],
        deterministic=True,
        augment=False,
        shuffle_buffer_size=None,
        number_images=None
    )

    # Setup Model
    coco_info = COCOAnalysis(cfg.train_annotation_path)
    num_classes = coco_info.get_num_classes()
    # Map from model index (0..N-1) to COCO category ID
    # Note: parse_example in training subtracts 1 from category_id.
    # If training IDs are 0..N-1, and COCO IDs are 1..N, we need to map back.
    # Let's assume strict mapping: dataset ID `i` corresponds to `sorted(category_ids)[i]`.
    category_ids_map = coco_info.get_category_ids()

    print(f"Number of classes: {num_classes}")

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

    # Load Checkpoint
    checkpoint_dir = cfg.test_model_path
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Restored model from {manager.latest_checkpoint}")
    else:
        print("No checkpoint found!")
        return

    # Inference and formatting
    results = []
    print("Starting inference...")
    start_time = time.time()

    @tf.function
    def predict_step(img):
        return model(img, training=False)

    count = 0
    # dataset yields: (image_resized, cat_ids, mask_targets, image_id, original_height, original_width)
    for images, _, _, image_ids, original_heights, original_widths in dataset:
        # Batch size is 1
        img_h = original_heights[0]
        img_w = original_widths[0]
        img_id = int(image_ids[0])

        # Predict
        pred_logits, pred_masks, _ = predict_step(images)

        # Post-process
        # pred_logits: [1, Q, K+1]
        # pred_masks: [1, Q, Hq, Wq]

        scores = tf.nn.softmax(pred_logits, axis=-1)[0, :, 1:]  # [Q, K] (exclude background)
        masks_raw = pred_masks[0]  # [Q, Hq, Wq]

        # For each query, find best class
        scores_max = tf.reduce_max(scores, axis=-1)  # [Q]
        labels = tf.argmax(scores, axis=-1)  # [Q]

        # Filter by threshold (using config threshold or default 0.05 for eval)
        keep = scores_max > 0.05

        scores_keep = scores_max[keep]
        labels_keep = labels[keep]
        masks_keep = masks_raw[keep]

        if tf.size(scores_keep) == 0:
            count += 1
            continue

        # Resize masks to original image dimensions
        # Expand for resize: [N, H, W, 1]
        masks_keep_exp = tf.expand_dims(masks_keep, -1)

        # Resize using bilinear then threshold, or nearest.
        # Bilinear is usually better for smooth scaling then threshold.
        masks_resized = tf.image.resize(
            masks_keep_exp,
            (img_h, img_w),
            method='bilinear'
        )
        masks_resized = tf.squeeze(masks_resized, -1)

        # Binary mask
        masks_binary = tf.cast(masks_resized > 0.5, tf.uint8).numpy()

        scores_keep_np = scores_keep.numpy()
        labels_keep_np = labels_keep.numpy()

        # Convert to RLE
        # pycocotools expects fortran order (column-major) for encoding
        for k in range(len(scores_keep_np)):
            label_idx = int(labels_keep_np[k])
            category_id = int(category_ids_map[label_idx])
            score = float(scores_keep_np[k])
            mask = masks_binary[k]  # [H, W]

            # Encode
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # Make JSON serializable

            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": rle,
                "score": score
            })

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")

    print(f"Inference finished in {time.time() - start_time:.2f}s")

    if not results:
        print("No predictions generated!")
        return

    # Save results to JSON
    # It's good practice to save intermediate results
    res_file = "predictions.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
    print(f"Predictions saved to {res_file}")

    # Evaluate
    print("Evaluating using COCOeval...")

    # Load GT
    # We use the annotation file from config
    ann_file = cfg.val_annotation_path
    coco_gt = COCO(ann_file)
    # Reassign category IDs to match training data
    reassign_category_ids(coco_gt)

    # Load DT
    coco_dt = coco_gt.loadRes(res_file)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()