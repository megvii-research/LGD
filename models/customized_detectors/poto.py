# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
# Modified from cvpods (https://github.com/Megvii-BaseDetection/cvpods)
# Copyright (c) Megvii, Inc. All Rights Reserved
# ------------------------------------------------------------------------------
from .build import CUSTOMIZED_DETECTORS_REGISTRY
from detectron2.modeling import detector_postprocess
from .thirdparty_heads import POTO
from typing import Dict, List, Tuple
from torch import Tensor, nn
import torch

@CUSTOMIZED_DETECTORS_REGISTRY.register()
class POTOCT(POTO):
    """
    Customize cvpods implemented `FCOS` forward api for convenient manipulation
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        #NOTE: separate fpn and backbone
        self.fpn = self.backbone
        self.raw_backbone = self.fpn.bottom_up
        self.fpn.bottom_up = nn.Sequential()

    def predict(self, features):
        box_cls, box_delta = self.head(features)
        shifts = self.shift_generator(features)
        return shifts, box_cls, box_delta


    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        """
        images = self.preprocess_image(batched_inputs)
        raw_features = self.raw_backbone(images.tensor)
        features = self.fpn(raw_features)

        features = [features[f] for f in self.in_features]
        shifts, box_cls, box_delta = self.predict(features)

        # wrap again in dict form
        features = dict(zip(self.in_features, features))

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_classes, gt_shifts_reg_deltas = \
                    self.get_ground_truth(shifts, gt_instances, box_cls, box_delta)
            losses = self.losses(gt_classes, gt_shifts_reg_deltas,
                    box_cls, box_delta)

            return losses, raw_features, features, images, \
                    (gt_classes, gt_shifts_reg_deltas)
        else:
            results = self.inference(box_cls, box_delta, shifts, images)
            processed_results = self.get_processed_results(results, batched_inputs, images)
            return processed_results, raw_features, features, images

    # separate the result processing as an api
    def get_processed_results(self, results, batched_inputs, images):
        if torch.jit.is_scripting():
            return results
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
