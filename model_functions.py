"""
Author: Pavel Timonin
Created: 2025-09-28
Description: This script contains classes and functions for Mask2Former model building.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense,
    Layer,
)
from tensorflow.keras.models import Model
from pixel_decoder import MSDeformablePixelDecoder
from transformer_decoder import TransformerDecoder


# FPN-ResNet50 backbone

def get_resnet50_backbone(input_shape=(480, 480, 3)):
    """
    Builds a ResNet50 backbone and returns feature maps C2, C3, C4, C5.

    Args:
        input_shape (tuple): Input image shape as (height, width, channels). Defaults to (480, 480, 3).

    Returns:
        tf.keras.Model: Model with inputs=image tensor and outputs=[c2, c3, c4, c5] feature maps.
    """
    backbone = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    backbone.trainable = True

    c2_output = backbone.get_layer("conv2_block3_out").output  # stride 4
    c3_output = backbone.get_layer("conv3_block4_out").output  # stride 8
    c4_output = backbone.get_layer("conv4_block6_out").output  # stride 16
    c5_output = backbone.get_layer("conv5_block3_out").output  # stride 32

    return Model(inputs=backbone.input, outputs=[c2_output, c3_output, c4_output, c5_output], name="resnet50_backbone")


# Mask2Former Head
class Mask2FormerHead(Layer):
    """
    Mask2Former-style head on top of pixel features.

    This layer processes flattened encoder features (memory) and per-pixel mask features
    to generate final class logits and mask predictions using a Transformer decoder.

    Args:
        num_classes (int): Number of object classes.
        num_queries (int): Number of object queries. Defaults to 100.
        d_model (int): Model dimension. Defaults to 256.
        num_decoder_layers (int): Number of decoder layers. Defaults to 6.
        num_heads (int): Number of attention heads. Defaults to 8.
        dim_feedforward (int): Feed-forward dimension. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        num_classes,
        num_queries=100,
        d_model=256,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        # Learnable query embeddings (content + positional, concatenated)
        self.query_embed = self.add_weight(
            name="query_embed",
            shape=(num_queries, d_model * 2),
            initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
            trainable=True,
        )
        # Transformer decoder (with mask attention)
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers, d_model=d_model,
            num_heads=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, name="transformer_decoder",
        )
        # Classification and mask embedding heads
        prior_prob = 0.01
        bias_value = -tf.math.log((1.0 - prior_prob) / prior_prob)
        self.class_embed = Dense(num_classes + 1, name="class_embed", bias_initializer=tf.keras.initializers.Constant(bias_value))
        self.mask_embed = tf.keras.Sequential([
            Dense(d_model, activation="relu"), Dense(d_model)
        ], name="mask_embed_mlp")

    def call(self, memory_list, memory_pos_list, decoder_shapes, mask_features, training=False):
        """
        Processes encoder features through transformer decoder to generate class and mask predictions.

        Args:
            memory_list (list): List of encoder feature maps (flattened) [B, S_l, C] for each feature level.
            memory_pos_list (list): List of positional encodings [B, S_l, C] for each feature level.
            decoder_shapes (tf.Tensor): Tensor [L, 2] giving (H_l, W_l) for each feature level in memory_list.
            mask_features (tf.Tensor): Low-level feature map [B, Hm, Wm, C] for computing masks (e.g., backbone output at 1/4 resolution).
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - pred_logits (tf.Tensor): Final class scores [B, num_queries, num_classes+1] for the last decoder layer.
                - pred_masks (tf.Tensor): Final predicted mask logits [B, num_queries, Hm, Wm] for the last decoder layer.
                - aux_outputs (list): List of dicts with intermediate predictions for auxiliary losses (one per decoder layer).
        """
        # Prepare query embeddings
        B = tf.shape(mask_features)[0]
        query_embed = tf.tile(self.query_embed[tf.newaxis, :, :], [B, 1, 1])
        query_content, query_pos = tf.split(query_embed, 2, axis=-1)
        tgt = query_content

        # Run transformer decoder
        decoder_outputs = self.decoder(
            tgt=tgt,
            memory_list=memory_list,
            memory_pos_list=memory_pos_list,
            decoder_shapes=decoder_shapes,
            query_pos=query_pos,
            mask_features=mask_features,
            mask_embed_fn=self.mask_embed,
            training=training,
        )

        # The decoder may include an initial state in intermediate_states; the last element is the final output
        if isinstance(decoder_outputs, list):
            intermediate_outputs = decoder_outputs
            final_output = intermediate_outputs[-1]  # [B, Q, C] from last decoder layer
        else:
            # If decoder is implemented to directly return final output
            intermediate_outputs = None
            final_output = decoder_outputs

        # Compute class and mask predictions
        all_class_logits = []
        all_mask_logits = []
        if intermediate_outputs is not None:
            for dec_out in intermediate_outputs:
                class_logits = self.class_embed(dec_out, training=training)
                mask_embed = self.mask_embed(dec_out, training=training)
                mask_logits = tf.einsum('bqc,bhwc->bqhw', mask_embed, mask_features)
                all_class_logits.append(class_logits)
                all_mask_logits.append(mask_logits)
        else:
            class_logits = self.class_embed(final_output, training=training)
            mask_embed = self.mask_embed(final_output, training=training)
            mask_logits = tf.einsum('bqc,bhwc->bqhw', mask_embed, mask_features)
            all_class_logits.append(class_logits)
            all_mask_logits.append(mask_logits)

        # Final predictions and auxiliary outputs
        all_class_logits = tf.stack(all_class_logits, axis=1)
        all_mask_logits = tf.stack(all_mask_logits, axis=1)
        pred_logits = all_class_logits[:, -1, ...]
        pred_masks = all_mask_logits[:, -1, ...]

        # Auxiliary outputs for training
        aux_outputs = [
            {"pred_logits": all_class_logits[:, i, ...], "pred_masks": all_mask_logits[:, i, ...]}
            for i in range(all_class_logits.shape[1] - 1)
        ]

        return pred_logits, pred_masks, aux_outputs


# Full Mask2Former Model

class Mask2FormerModel(tf.keras.Model):
    """
    TensorFlow implementation of a Mask2Former-style model.

    Architecture:
        image -> ResNet50 (C2, C3, C4, C5)
              -> PixelDecoder (MSDeformablePixelDecoder)
              -> TransformerDecoder (Mask2FormerHead)
              -> class logits + mask logits

    This is a *model-only* implementation: no loss functions or matching.

    Args:
        input_shape (tuple): Input image shape. Defaults to (480, 480, 3).
        transformer_input_channels (int): Channel dimension for transformer inputs. Defaults to 256.
        num_classes (int): Number of object classes. Defaults to 80.
        num_queries (int): Number of object queries. Defaults to 100.
        num_decoder_layers (int): Number of decoder layers. Defaults to 6.
        num_heads (int): Number of attention heads. Defaults to 8.
        dim_feedforward (int): Feed-forward dimension. Defaults to 1024.
        dropout (float): Dropout rate. Defaults to 0.1.
        name (str): Model name. Defaults to "mask2former_tf".
        **kwargs: Additional keyword arguments for the base Model class.
    """

    def __init__(
        self,
        input_shape=(480, 480, 3),
        transformer_input_channels=256,
        num_classes=80,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        name="mask2former_tf",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer_input_channels = transformer_input_channels

        # ResNet backbone
        self.backbone = get_resnet50_backbone(
            input_shape=input_shape
        )

        # Pixel decoder
        self.pixel_decoder = MSDeformablePixelDecoder(
            d_model=transformer_input_channels,
            num_feature_levels=4,  # C2, C3, C4, C5
            transformer_num_feature_levels=3,  # deformable encoder uses C3, C4, C5
            num_encoder_layers=6,  # or any value you prefer
            n_heads=num_heads,
            n_points=4,  # typical value in Mask2Former
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            name="pixel_decoder",
        )

        # Mask2Former head
        self.mask2former_head = Mask2FormerHead(
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=transformer_input_channels,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            name="mask2former_head",
        )

    def call(self, inputs, training=False):
        """
        Forward pass through the complete Mask2Former model.

        Args:
            inputs (tf.Tensor): Input images of shape [B, H, W, 3].
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - pred_logits (tf.Tensor): Class predictions [B, num_queries, num_classes+1].
                - pred_masks (tf.Tensor): Mask predictions [B, num_queries, Hm, Wm].
                - aux_outputs (list): Auxiliary outputs for deep supervision.
        """
        c2, c3, c4, c5 = self.backbone(inputs, training=training)

        memory_list, memory_pos_list, decoder_shapes, mask_features = self.pixel_decoder(
            [c2, c3, c4, c5],
            training=training,
        )

        pred_logits, pred_masks, aux_outputs = self.mask2former_head(
            memory_list=memory_list,
            memory_pos_list=memory_pos_list,
            decoder_shapes=decoder_shapes,
            mask_features=mask_features,
            training=training,
        )

        return pred_logits, pred_masks, aux_outputs