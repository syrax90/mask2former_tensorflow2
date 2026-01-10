"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains classes and functions for Mask2Former model building.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Conv2D,
    UpSampling2D,
    Add,
    Dense,
    Layer,
)
from tensorflow.keras.models import Model
from pixel_decoder import MSDeformablePixelDecoder
from transformer_decoder import TransformerDecoder


# ---------------------------------------------------------------------
# FPN-ResNet50 backbone
# ---------------------------------------------------------------------

def build_fpn_resnet50(input_shape=(480, 480, 3), ouput_kernels_number=256):
    """
    Builds a Feature Pyramid Network (FPN) on top of a ResNet50 backbone.

    Returns:
        tf.keras.Model with inputs=image tensor and outputs=[p2, p3, p4, p5].
    """
    backbone = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    backbone.trainable = False

    # ResNet feature maps
    c2_output = backbone.get_layer("conv2_block3_out").output  # stride 4
    c3_output = backbone.get_layer("conv3_block4_out").output  # stride 8
    c4_output = backbone.get_layer("conv4_block6_out").output  # stride 16
    c5_output = backbone.get_layer("conv5_block3_out").output  # stride 32

    # Lateral connections
    p5 = Conv2D(256, (1, 1), padding="same", name="fpn_lateral_c5")(c5_output)
    p4 = Add(name="fpn_p4_add")(
        [
            UpSampling2D(name="fpn_p5_upsample")(p5),
            Conv2D(256, (1, 1), padding="same", name="fpn_lateral_c4")(c4_output),
        ]
    )
    p3 = Add(name="fpn_p3_add")(
        [
            UpSampling2D(name="fpn_p4_upsample")(p4),
            Conv2D(256, (1, 1), padding="same", name="fpn_lateral_c3")(c3_output),
        ]
    )
    p2 = Add(name="fpn_p2_add")(
        [
            UpSampling2D(name="fpn_p3_upsample")(p3),
            Conv2D(256, (1, 1), padding="same", name="fpn_lateral_c2")(c2_output),
        ]
    )

    # Smoothing (3×3 convs)
    p5 = Conv2D(
        ouput_kernels_number, (3, 3), padding="same", name="fpn_p5_smooth"
    )(p5)
    p4 = Conv2D(
        ouput_kernels_number, (3, 3), padding="same", name="fpn_p4_smooth"
    )(p4)
    p3 = Conv2D(
        ouput_kernels_number, (3, 3), padding="same", name="fpn_p3_smooth"
    )(p3)
    p2 = Conv2D(
        ouput_kernels_number, (3, 3), padding="same", name="fpn_p2_smooth"
    )(p2)

    return Model(inputs=backbone.input, outputs=[p2, p3, p4, p5], name="fpn_resnet50")


# ---------------------------------------------------------------------
# Mask2Former Head (class + mask prediction)
# ---------------------------------------------------------------------
class Mask2FormerHead(Layer):
    """
    Mask2Former-style head on top of pixel features.

    Inputs:
        memory:        (B, HW, C)    flattened encoder/pixel features
        mask_features: (B, Hm, Wm, C) per-pixel features used to generate masks

    Outputs (all Tensors, graph-friendly):
        pred_logits:      (B, num_queries, num_classes + 1)
        pred_masks:       (B, num_queries, Hm, Wm)
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
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        # Transformer decoder (with mask attention)
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers, d_model=d_model,
            num_heads=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, name="transformer_decoder",
        )
        # Classification head and Mask embedding head
        prior_prob = 0.01
        bias_value = -tf.math.log((1.0 - prior_prob) / prior_prob)

        self.class_embed = Dense(num_classes + 1, name="class_embed", bias_initializer=tf.keras.initializers.Constant(bias_value))  # (includes "no object")
        self.mask_embed = tf.keras.Sequential([
            Dense(d_model, activation="relu"), Dense(d_model)
        ], name="mask_embed_mlp")

    def call(self, memory_list, memory_pos_list, decoder_shapes, mask_features, training=False):
        """
        Args:
            memory_list: list of encoder feature maps (flattened) [B, S_l, C] for each feature level.
            memory_pos_list: list of positional encodings [B, S_l, C] for each feature level.
            decoder_shapes: Tensor [L, 2] giving (H_l, W_l) for each feature level in memory_list.
            mask_features: [B, Hm, Wm, C] low-level feature map for computing masks (e.g., backbone output at 1/4 resolution).
        Returns:
            pred_logits: [B, num_queries, num_classes+1] final class scores (for the last decoder layer).
            pred_masks:  [B, num_queries, Hm, Wm] final predicted mask logits (for the last decoder layer).
            aux_outputs (optional): list of dicts with intermediate predictions for auxiliary losses (one per decoder layer).
        """
        B = tf.shape(mask_features)[0]
        # Prepare query embeddings
        query_embed = tf.tile(self.query_embed[tf.newaxis, :, :], [B, 1, 1])  # [B, Q, 2C]
        query_content, query_pos = tf.split(query_embed, 2, axis=-1)  # each [B, Q, C]
        tgt = query_content  # initial query content (to be updated by decoder)

        # Run the transformer decoder (with masked attention)
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

        # Compute class and mask predictions for each decoder output
        all_class_logits = []
        all_mask_logits = []
        if intermediate_outputs is not None:
            for dec_out in intermediate_outputs:
                class_logits = self.class_embed(dec_out, training=training)  # [B, Q, num_classes+1]
                mask_embed = self.mask_embed(dec_out, training=training)  # [B, Q, C]
                mask_logits = tf.einsum('bqc,bhwc->bqhw', mask_embed, mask_features)  # [B, Q, Hm, Wm]
                all_class_logits.append(class_logits)
                all_mask_logits.append(mask_logits)
        else:
            # If no intermediate list (should not happen in this design), just compute from final_output
            class_logits = self.class_embed(final_output, training=training)
            mask_embed = self.mask_embed(final_output, training=training)
            mask_logits = tf.einsum('bqc,bhwc->bqhw', mask_embed, mask_features)
            all_class_logits.append(class_logits)
            all_mask_logits.append(mask_logits)

        # Stack outputs: shape [B, Layers, Q, ...]
        all_class_logits = tf.stack(all_class_logits, axis=1)  # [B, L_dec, Q, num_classes+1]
        all_mask_logits = tf.stack(all_mask_logits, axis=1)  # [B, L_dec, Q, Hm, Wm]

        # Final predictions from the last layer (for inference)
        pred_logits = all_class_logits[:, -1, ...]  # [B, Q, num_classes+1]
        pred_masks = all_mask_logits[:, -1, ...]  # [B, Q, Hm, Wm]

        # (Optional) auxiliary outputs for intermediate layers (for training loss)
        aux_outputs = [
            {"pred_logits": all_class_logits[:, i, ...], "pred_masks": all_mask_logits[:, i, ...]}
            for i in range(all_class_logits.shape[1] - 1)  # exclude the last one, as it's the final output
        ]

        return pred_logits, pred_masks, aux_outputs


# ---------------------------------------------------------------------
# Full Mask2Former-style Keras Model
# ---------------------------------------------------------------------

class Mask2FormerModel(tf.keras.Model):
    """
    TensorFlow implementation of a Mask2Former-style model.

    Architecture:
        image -> FPN-ResNet50 -> mask feature map (stride 4)
              -> transformer-based Mask2Former head
              -> class logits + mask logits

    This is a *model-only* implementation: no loss functions or matching.
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

        # 1) FPN-ResNet backbone
        self.backbone = build_fpn_resnet50(
            input_shape=input_shape,
            ouput_kernels_number=transformer_input_channels,
        )

        self.pixel_decoder = MSDeformablePixelDecoder(
            d_model=transformer_input_channels,
            num_feature_levels=4,  # p2, p3, p4, p5
            transformer_num_feature_levels=3,  # deformable encoder uses p3, p4, p5
            num_encoder_layers=6,  # or any value you prefer
            n_heads=num_heads,
            n_points=4,  # typical value in Mask2Former
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            name="pixel_decoder",
        )

        # 3) Mask2Former head
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
        p2, p3, p4, p5 = self.backbone(inputs, training=training)

        memory_list, memory_pos_list, decoder_shapes, mask_features = self.pixel_decoder(
            [p2, p3, p4, p5],
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
