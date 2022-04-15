# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
from .build import CUSTOMIZED_DETECTORS_REGISTRY
from detectron2.modeling import RetinaNet, detector_postprocess
from typing import Dict, List, Tuple
from torch import Tensor, nn
import torch

def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

@CUSTOMIZED_DETECTORS_REGISTRY.register()
class RetinaNetCT(RetinaNet):
    """
    Customize detectron2 official `RetinaNet` forward api for convenient manipulation
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        #NOTE: separate fpn and backbone
        self.fpn = self.backbone
        self.raw_backbone = self.fpn.bottom_up
        self.fpn.bottom_up = nn.Sequential()

    def predict(self, features):
        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        return anchors, pred_logits, pred_anchor_deltas

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        """
        images = self.preprocess_image(batched_inputs)

        # separate feature extraction
        #features = self.backbone(images.tensor)
        raw_features = self.raw_backbone(images.tensor)
        features = self.fpn(raw_features)

        features = [features[f] for f in self.head_in_features]
        anchors, pred_logits, pred_anchor_deltas = self.predict(features)

        # wrap again in dict form
        features = dict(zip(self.head_in_features, features))

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses, raw_features, features, images, (gt_labels, gt_boxes)
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
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
