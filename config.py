"""
Author: Pavel Timonin
Created: 2025-09-28
Description: This script contains configurations.
"""


class Mask2FormerConfig(object):
    """
    Configuration class for Mask2Former model parameters and paths.

    This class holds all necessary configuration settings for data loading,
    model training, and testing, including file paths, image dimensions,
    hyperparameters, and dataset options.
    """
    def __init__(self):
        self.coco_root_path = '/path/to/your/coco/root/directory'
        self.train_annotation_path = f'{self.coco_root_path}/annotations/instances_train2017.json'
        self.tfrecord_dataset_directory_path = 'path/to/tfrecords/train/directory'  # Path to TFRecord dataset directory
        self.number_images=None  # Restriction for dataset. Set None to get rid of the restriction

        # Image parameters
        self.img_height = 480
        self.img_width = 480

        # If load_previous_model = True: load the previous model weights.
        self.load_previous_model = False
        self.lr = 0.0001
        self.batch_size = 16
        # If load_previous_model = True, the code will look for the latest checkpoint in this directory or use this path if it is a specific checkpoint file.
        self.model_path = './checkpoints'  # example for specific checkpoint: self.model_path = './checkpoints/ckpt-5'

        # Save the model weights every save_iter epochs:
        self.save_iter = 1
        self.approx_coco_train_size = 118287
        # Number of epochs
        self.epochs = 100

        # Testing configuration
        self.test_model_path = './checkpoints'  # example for specific checkpoint: self.test_model_path = './checkpoints/ckpt-5'
        self.score_threshold = 0.5

        # Accumulation mode
        self.use_gradient_accumulation_steps = False
        self.accumulation_steps = 8

        # Dataset options
        self.tfrecord_test_path = 'path/to/tfrecords/test/directory'  # Path to TFRecord test dataset directory. Used for mAP calculation.
        self.image_scales = [0.25]
        self.augment = True
        self.shuffle_buffer_size = 4096  # TFRecord dataset shuffle buffer size. Set to None to disable shuffling
        self.warmup_steps = 10000
