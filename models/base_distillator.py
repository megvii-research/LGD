# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .customized_detectors import build_customized_detector
from .adapters import build_adapter
from abc import abstractmethod

class BaseDistillator(nn.Module):
    """
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.norm_stu = nn.InstanceNorm2d(256, affine=False)
        self.norm_tea = nn.InstanceNorm2d(256, affine=False)

        self.student, self.teacher = build_customized_detector(cfg)

        self.coef = cfg.MODEL.DISTILLATOR.LAMBDA


        self.add_bg_box = cfg.MODEL.DISTILLATOR.TEACHER.ADD_CONTEXT_BOX

        self.adapter = nn.ModuleDict({'distill': build_adapter(cfg)})


    def distill_loss(self, features, images, batched_inputs, batchified_inside_masks, inst_labels):
        losses = dict()
        losses["loss_distill"] = self.distill(features, images, batched_inputs, batchified_inside_masks, inst_labels)
        return losses

    def distill(self, features, images, batched_inputs, batchified_inside_masks, fg_labels):
        '''
        Input:
            features        : Dict of List of Tensors
            images          : ImageList
            batched_inputs  : Raw inputs
            batchified_inside_masks: F x B x (Ni, HiWi)
            fg_labels:             B x (Ni-1, ) if add_bg_box=True else B x (Ni,)
        Output:
            loss: Tensor
        '''
        keys = sorted(features['stu'].keys() & features['tea'].keys())
        bs = features['tea'][keys[0]].size(0)

        tea_features = [features['tea'][k] for k in keys]
        stu_features = [features['stu'][k] for k in keys]

        #NOTE: student detach or not depends, teacher always detach
        if self.distill_flag == 0:
            stu_features = [feat.detach() for feat in stu_features]

        tea_features = [feat.detach() for feat in tea_features]

        stu_features = [self.adapter['distill'](k) for k in stu_features]

        stu_features = [self.norm_stu(f) for f in stu_features]
        tea_features = [self.norm_tea(feat) for feat in tea_features]

        stu_features = torch.cat([f.view(bs, -1) for f in stu_features], dim=1)
        tea_features = torch.cat([t.view(bs, -1) for t in tea_features], dim=1)
        return self.coef * F.mse_loss(tea_features, stu_features)


    @abstractmethod
    def forward(self, batched_inputs, **kwargs):
        pass

    @abstractmethod
    def forward_student(self, batched_inputs, **kwargs):
        pass

    @abstractmethod
    def forward_teacher(self, batched_inputs, **kwargs):
        pass
