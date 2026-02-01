"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script performs the main actions for the training process.
"""


import os
import logging
import tensorflow as tf
import tensorflow.keras.layers as layers
from coco_dataset_optimized import create_coco_tfrecord_dataset, get_classes
from config import Mask2FormerConfig
from model_functions import Mask2FormerModel
from loss import compute_multiscale_loss

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

tf.keras.backend.clear_session()

# Enable dynamic memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"



def train_one_epoch(model, dataset, optimizer, num_classes):
    """
    Run one training epoch without gradient accumulation.

    Iterates over the dataset once, computes loss, applies gradients, and prints
    per-step metrics.

    Args:
        model (tf.keras.Model): Mask2Former model whose forward pass returns
            `(class_outputs, mask_outputs, mask_feat)` for various scales/features.
        dataset (tf.data.Dataset): Yields triples `(images, cate_target, mask_target)` per step.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to update weights.
        num_classes (int): Number of object classes (background excluded).

    Returns:
        None
    """
    for step, (images, cate_target, mask_target)  in enumerate(dataset):
        total_loss, cate_loss, dice_loss, mask_loss = train_one_step(model, images, cate_target, mask_target, optimizer, num_classes)
        print("Step ", step, ": ", "total=", total_loss.numpy(), ", cate=", cate_loss.numpy(), ", dice=", dice_loss.numpy(), ", mask=", mask_loss.numpy())

@tf.function(experimental_relax_shapes=True)
def train_one_step(model, images, cate_target, mask_target, optimizer, num_classes):
    """
    Perform a single optimization step (no accumulation).

    Runs a forward pass, computes multiscale Mask2Former losses, backpropagates, and
    applies gradients.

    Args:
        model (tf.keras.Model): Mask2Former model producing class outputs, mask outputs and features.
        images (tf.Tensor): float32 input images, `[B, H, W, 3]`.
        cate_target (tf.Tensor): Class indices per grid cell (a.k.a. `class_target`), `[B, sum(S_i^2)]`.
        mask_target (tf.Tensor): GT masks aligned to grid cells, `[B, H, W, sum(S_i^2)]`.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.
        num_classes (int): Number of classes (background excluded).

    Returns:
        tuple: A tuple containing:
            - total_loss (tf.Tensor): Scalar total loss.
            - cate_loss (tf.Tensor): Scalar classification (focal) loss.
            - dice_loss (tf.Tensor): Scalar mask (Dice) loss.
            - mask_loss (tf.Tensor): Scalar mask (CE) loss.
    """
    with tf.GradientTape() as tape:
        pred_logits, pred_masks, aux_outputs = model(images, training=True)
        total_loss, cate_loss, dice_loss, mask_loss = compute_multiscale_loss(
            pred_logits, pred_masks,
            cate_target, mask_target,
            aux_outputs=aux_outputs,
            num_classes=num_classes)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, cate_loss, dice_loss, mask_loss

def accumulate_one_step(model,
                        images,
                        cate_target,
                        mask_target,
                        num_classes,
                        accum_grads):
    """
    Accumulate gradients for one mini-batch (no optimizer step).

    Computes Mask2Former multiscale losses and adds gradients into preallocated
    buffers to enable gradient accumulation across multiple steps.

    Args:
        model (tf.keras.Model): Mask2Former model producing class outputs and mask outputs.
        images (tf.Tensor): float32 `[B, H, W, 3]`.
        cate_target (tf.Tensor): Class indices `[B, sum(S_i^2)]`.
        mask_target (tf.Tensor): GT masks `[B, H, W, sum(S_i^2)]`.
        num_classes (int): Number of classes (background excluded).
        accum_grads (list): Zero-initialized gradient buffers (tf.Variable),
            one per entry in `model.trainable_variables`; same shapes/dtypes.

    Returns:
        tuple: A tuple containing:
            - total_l (tf.Tensor): Scalar total loss.
            - cate_l (tf.Tensor): Scalar classification loss.
            - dice_loss (tf.Tensor): Scalar mask (Dice) loss.
            - mask_l (tf.Tensor): Scalar mask loss.
    """
    with tf.GradientTape() as tape:
        pred_logits, pred_masks, aux_outputs = model(images, training=True)
        total_l, cate_l, dice_loss, mask_l = compute_multiscale_loss(
            pred_logits, pred_masks,
            cate_target, mask_target,
            aux_outputs=aux_outputs,
            num_classes=num_classes
        )

    grads = tape.gradient(total_l, model.trainable_variables)

    # Add to buffers
    for g_acc, g in zip(accum_grads, grads):
        if g is not None:
            g_acc.assign_add(g)

    return total_l, cate_l, dice_loss, mask_l

@tf.function(experimental_relax_shapes=True)
def train_one_epoch_accumulated_mode(model,
                    dataset,
                    optimizer,
                    num_classes,
                    accumulation_steps,
                    accum_grads,
                    accum_counter,
                    global_step):
    """
    Run one epoch with gradient accumulation using preallocated buffers.

    For each batch, compute Mask2Former multiscale losses and add gradients to
    `accum_grads`. Apply an optimizer step every `accumulation_steps` by
    dividing buffered grads by `accumulation_steps` and resetting buffers.
    Any leftover buffered grads at epoch end are cleared (no update).

    Args:
        model (tf.keras.Model): Mask2Former model producing class outputs, mask outputs and features.
        dataset (tf.data.Dataset): Yields `(images, cate_target, mask_target)`.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.
        num_classes (int): Number of classes (background excluded).
        accumulation_steps (int): Number of mini-batches to accumulate.
        accum_grads (list): Gradient buffers (tf.Variable), zeros, same shapes as `model.trainable_variables`.
        accum_counter (tf.Variable): int32 counter tracking steps since last apply.
        global_step (tf.Variable): int32 counter of total steps in the epoch.

    Returns:
        None
    """

    # Helper: apply optimizer and reset buffers
    def _apply_and_reset(denominator):
        scaled = [g / tf.cast(denominator, g.dtype) for g in accum_grads]
        optimizer.apply_gradients(zip(scaled, model.trainable_variables))
        for g in accum_grads:
            g.assign(tf.zeros_like(g))
        accum_counter.assign(0)

    def _reset_counter():
        for g in accum_grads:
            g.assign(tf.zeros_like(g))
        accum_counter.assign(0)

    for images, cate_target, mask_target in dataset:
        tot, cat, dice, msk = accumulate_one_step(
            model, images, cate_target, mask_target, num_classes, accum_grads)

        accum_counter.assign_add(1)
        global_step.assign_add(1)

        tf.cond(accum_counter == accumulation_steps,
                lambda: _apply_and_reset(accumulation_steps),
                lambda: None)

        tf.print("step", global_step,
                 ": total=", tot,
                 "cate=", cat,
                 "dice=", dice,
                 "mask=", msk)

    tf.cond(accum_counter > 0,
            lambda: _reset_counter(),
            lambda: None)

def run_one_epoch_accumulated_mode(model,
                  dataset,
                  optimizer,
                  num_classes,
                  accumulation_steps=8):
    """
    Convenience wrapper to run one epoch with gradient accumulation.

    Allocates zero-initialized gradient buffers and integer counters, then
    calls :func:`train_one_epoch_accumulated_mode`.

    Args:
        model (tf.keras.Model): Mask2Former model producing class outputs and mask outputs.
        dataset (tf.data.Dataset): Yields `(images, cate_target, mask_target)`.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.
        num_classes (int): Number of classes (background excluded).
        accumulation_steps (int, optional): Steps to accumulate before applying updates. Defaults to `8`.

    Returns:
        None
    """
    # Gradient buffers (one per trainable weight)
    accum_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False)
        for v in model.trainable_variables
    ]

    # Counters
    accum_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    global_step   = tf.Variable(0, dtype=tf.int32, trainable=False)

    # Run the compiled graph
    train_one_epoch_accumulated_mode(model,
                    dataset,
                    optimizer,
                    num_classes,
                    accumulation_steps,
                    accum_grads,
                    accum_counter,
                    global_step)


if __name__ == '__main__':
    cfg = Mask2FormerConfig()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    batch_size = cfg.batch_size
    img_height, img_width = cfg.img_height, cfg.img_width

    # Create a model instance
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

    # Learning Rate Schedule
    # Estimate total steps for CosineDecay
    if cfg.number_images:
        total_steps = (cfg.number_images // cfg.batch_size) * cfg.epochs
    else:
        # If number_images is not set, we assume the full COCO 2017 training set size (~118k images).
        # This provides a reasonable 'decay_steps' scale for the scheduler even if the exact number varies slightly.
        steps_per_epoch = cfg.approx_coco_train_size // cfg.batch_size
        total_steps = steps_per_epoch * cfg.epochs

    warmup_steps = cfg.warmup_steps
    initial_learning_rate = cfg.lr

    # Scheduler
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=total_steps - warmup_steps,
        alpha=0.01
    )


    class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
        """
        Learning rate schedule with a warmup period.

        Linearly increases the learning rate from 0 to `target_lr` over `warmup_steps`,
        then follows the `decay_schedule`.

        Args:
            warmup_steps (int): Number of steps for the warmup phase.
            target_lr (float): Target learning rate after warmup.
            decay_schedule (tf.keras.optimizers.schedules.LearningRateSchedule):
                Schedule to follow after warmup.
        """
        def __init__(self, warmup_steps, target_lr, decay_schedule):
            super(WarmUp, self).__init__()
            self.warmup_steps = warmup_steps
            self.target_lr = target_lr
            self.decay_schedule = decay_schedule

        def __call__(self, step):
            return tf.cond(
                step < self.warmup_steps,
                lambda: self.target_lr * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)),
                lambda: self.decay_schedule(step - self.warmup_steps)
            )

        def get_config(self):
            return {
                "warmup_steps": self.warmup_steps,
                "target_lr": self.target_lr,
                "decay_schedule": self.decay_schedule
            }


    learning_rate = WarmUp(warmup_steps, initial_learning_rate, lr_schedule)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.05)

    # Checkpoint mechanism
    epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=epoch_var)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=10)

    if cfg.load_previous_model:
        checkpoint_path = cfg.model_path
        if os.path.isdir(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

        if checkpoint_path:
            checkpoint.restore(checkpoint_path).expect_partial()
            print(f"Restored from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Load previous model is True but no checkpoint found in {cfg.model_path}")
    else:
        print("Starting from scratch.")

    previous_epoch = int(epoch_var.numpy())

    if previous_epoch > cfg.epochs:
        print(f'The model is trained {previous_epoch} epochs already while configuration assumes {cfg.epochs} epochs.')
        exit(0)

    # Form COCO dataset
    ds = create_coco_tfrecord_dataset(
        train_tfrecord_directory=cfg.tfrecord_dataset_directory_path,
        target_size=(img_height, img_width),
        batch_size=cfg.batch_size,
        scale=cfg.image_scales[0],
        augment=cfg.augment,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        number_images=cfg.number_images)

    # Training loop
    print("Starting training...")
    for epoch in range(previous_epoch + 1, cfg.epochs + 1):
        print(f"Starting epoch {epoch}:")
        if cfg.use_gradient_accumulation_steps:
            run_one_epoch_accumulated_mode(model, ds, optimizer, num_classes, accumulation_steps=cfg.accumulation_steps)
        else:
            train_one_epoch(model, ds, optimizer, num_classes)

        # Update epoch variable
        epoch_var.assign(epoch)

        if epoch % cfg.save_iter == 0:
            save_path = manager.save()
            logger.info('Saved checkpoint for epoch {}: {}'.format(epoch, save_path))
    print("Done!")
