# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
from .build import ADAPTERS_REGISTRY
from torch import nn

@ADAPTERS_REGISTRY.register()
class SequentialConvs(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.adapter =  nn.Sequential(*[nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1)])
    def forward(self, x):
        return self.adapter(x)
