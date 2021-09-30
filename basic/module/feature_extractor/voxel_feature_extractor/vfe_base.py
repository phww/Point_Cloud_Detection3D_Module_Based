#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 下午4:32
# @Author : PH
# @Version：V 0.1
# @File : vfe_base.py
# @desc :
import torch.nn as nn


class VFEBase(nn.Module):

    def __init__(self, module_cfg):
        super(VFEBase, self).__init__()
        self.module_cfg = module_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError