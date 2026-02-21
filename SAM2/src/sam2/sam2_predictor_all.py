# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_predictor_single import SAM2ImagePredictorSingle
from sam2.utils.amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    MaskData,
)

class SAM2Predictor:
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        point_grids: Optional[List[np.ndarray]] = None,
        output_mode: str = "binary_mask",
        multimask_output: bool = True,
    ) -> None:
        """
        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          multimask_output (bool): Whether to output multimask at each point of the grid.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                0,
                1,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",

        ], f"Unknown output_mode {output_mode}."
 
        self.predictor = SAM2ImagePredictorSingle(
            model,
            max_hole_area=0,
            max_sprinkle_area=0,
        )
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.output_mode = output_mode
        self.multimask_output = multimask_output

    @torch.no_grad()
    def generate_step2(self, image: np.ndarray):
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           image_embedding, masks, mask_feature.
        """
        orig_size = image.shape[:2]
        # Crop the image and calculate embeddings
        # self.predictor.set_image(image)

        # Get points for this crop
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data.to_numpy()
        return data["masks"], data["mask_feature"]

    @torch.no_grad()
    def generate(self, image: np.ndarray):
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           image_embedding, masks, mask_feature.
        """
        orig_size = image.shape[:2]
        # Crop the image and calculate embeddings
        self.predictor.set_image(image)

        # Get points for this crop
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data.to_numpy()
        return data["masks"], data["mask_feature"]

    def _process_batch(
        self,
        points: np.ndarray,
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        # Run model on this batch
        points = torch.as_tensor(points, device=self.predictor.device)
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=orig_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, low_res_masks, mask_feature  = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            mask_feature = mask_feature
        )
        del masks
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        # Calculate and filter by stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        return data
    
