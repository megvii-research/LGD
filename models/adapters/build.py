# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
import torch
from detectron2.utils.registry import Registry

ADAPTERS_REGISTRY = Registry("ADAPTERS")
ADAPTERS_REGISTRY.__doc__ = ""

def build_adapter(cfg):
    meta_arch = cfg.MODEL.DISTILLATOR.ADAPTER.META_ARCH
    model = ADAPTERS_REGISTRY.get(meta_arch)(cfg)
    model = model.to(torch.device(cfg.MODEL.DEVICE))
    return model
