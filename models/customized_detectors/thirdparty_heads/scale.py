# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Taken from cvpods (https://github.com/Megvii-BaseDetection/cvpods)
# Copyright (c) Megvii, Inc. All Rights Reserved
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
