# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn

class STN(nn.Module):
    '''
    Implementation to https://arxiv.org/abs/1506.02025
    '''
    def __init__(self, k=64):
        super(STN, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.LayerNorm([64, 1], elementwise_affine=False)
        self.bn2 = nn.LayerNorm([128, 1], elementwise_affine=False)
        self.bn3 = nn.LayerNorm([1024, 1], elementwise_affine=False)
        self.bn4 = nn.LayerNorm([512], elementwise_affine=False)
        self.bn5 = nn.LayerNorm([256], elementwise_affine=False)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        #NOTE: Discard Identity matrix short-cut (since I do no apply the orthogonal regularization)
        #iden = torch.eye(self.k, device=x.device).unsqueeze(0).view(1, -1)
        #x = x + iden

        x = x.view(-1, self.k, self.k)
        return x
