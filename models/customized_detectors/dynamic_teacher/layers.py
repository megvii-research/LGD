# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch.nn as nn

def get_norm(channels, nr_groups=1, affine_flag=False):
    return nn.GroupNorm(num_groups=nr_groups, num_channels=channels, affine=affine_flag)

def get_MLP(nr_layers, channels, has_norm, has_relu=True, affine_flag=False):
    def _unit(with_norm, with_relu):
        tmp = list()
        tmp.append(nn.Linear(channels, channels))
        if with_norm:
            tmp.append(nn.LayerNorm([channels], elementwise_affine=affine_flag))
        if with_relu:
            tmp.append(nn.ReLU())
        return nn.Sequential(*tmp)
    layers = [_unit(has_norm, has_relu) for _ in range(nr_layers)]
    return nn.Sequential(*layers)


def get_CONVS(nr_layers, channels, has_norm, has_relu=True, nr_groups=1, affine_flag=False):
    def _unit(with_norm, with_relu):
        tmp = list()
        tmp.append(nn.Conv2d(channels, channels, 3, 1, 1))
        if with_norm:
            tmp.append(get_norm(channels, nr_groups, affine_flag))
        if with_relu:
            tmp.append(nn.ReLU())
        return nn.Sequential(*tmp)
    layers = [_unit(has_norm, has_relu) for _ in range(nr_layers)]
    return nn.Sequential(*layers)
