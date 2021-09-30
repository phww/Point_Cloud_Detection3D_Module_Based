#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 上午10:18
# @Author : PH
# @Version：V 0.1
# @File : mlp_vfe.py
# @desc :
import torch

from .vfe_base import VFEBase
import torch.nn as nn
from functools import partial


class MlpVFE(VFEBase):

    def __init__(self, module_cfg, **kwargs):
        super(MlpVFE, self).__init__(module_cfg)
        mlp_dims = module_cfg['mlp_dims']  # [32 64 64 128 1024]
        input_channels = module_cfg['input_channels']
        mlps = []
        # bn = partial(nn.BatchNorm1d)
        for out_channels in mlp_dims:
            mlps.append(nn.Linear(input_channels, out_channels))
            # mlps.append(bn(out_channels))
            mlps.append(nn.ReLU())
            input_channels = out_channels
        self.mlp = nn.Sequential(*mlps)

    def forward(self, batch_dick, **kwargs):
        x = batch_dick['voxels']
        # x = x.permute(0, 2, 1)  # B, N, C -> B, C, N
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        return x  # .permute(0, 2, 1)  # B, N, C

    def get_output_feature_dim(self):
        pass
