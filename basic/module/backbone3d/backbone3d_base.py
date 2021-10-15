#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/15 下午2:21
# @Author : PH
# @Version：V 0.1
# @File : backbone3d_base.py
# @desc :
import torch.nn as nn


class VoxelBackBone3D(nn.Module):

    def __init__(self, module_cfg, model_info_dict):
        super(VoxelBackBone3D, self).__init__()
        self.module_cfg = module_cfg
        self.model_info_dict = model_info_dict
        self.input_channels = model_info_dict['cur_point_feature_dims']
        self.grid_size = model_info_dict['grid_size']

    def forward(self, batch_dict):
        raise NotImplementedError

    @property
    def output_feature_dims(self):
        raise NotImplementedError

    @property
    def output_feature_size(self):
        raise NotImplementedError
