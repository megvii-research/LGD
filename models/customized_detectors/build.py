# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
import torch
from detectron2.utils.registry import Registry
from detectron2.modeling import META_ARCH_REGISTRY
import torch.nn as nn
CUSTOMIZED_DETECTORS_REGISTRY = Registry("CUSTOMIZED_DETECTORS")
CUSTOMIZED_DETECTORS_REGISTRY.__doc__ = ""

def get_model(cfg, meta_arch):
    model = CUSTOMIZED_DETECTORS_REGISTRY.get(meta_arch)(cfg)
    model = model.to(torch.device(cfg.MODEL.DEVICE))
    return model


class CustomModelWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = META_ARCH_REGISTRY.get(cfg.MODEL.DISTILLATOR.STUDENT.META_ARCH)(cfg)
        self.holder = [None]

        def register_hook(module, fea_in, fea_out):
            self.holder[0] = fea_out


        self.model.backbone.register_forward_hook()

    def forward(self, batched_inputs):
        out = self.model(batched_inputs)
        feat = self.holder[0]
        image = self.model.preprocess_image(batched_inputs)
        if self.training:
            return processed_results, raw_features, features, images


def build_customized_detector(cfg):
    student = get_model(cfg, cfg.MODEL.DISTILLATOR.STUDENT.META_ARCH)
    teacher = get_model(cfg, cfg.MODEL.DISTILLATOR.TEACHER.META_ARCH)
    return student, teacher
