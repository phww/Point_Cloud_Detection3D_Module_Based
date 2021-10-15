#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/15 下午2:43
# @Author : PH
# @Version：V 0.1
# @File : neck_base.py
# @desc :
import torch.nn as nn


class NECK(nn.Module):

    def __init__(self, module_cfg, model_info_dict):
        super(NECK, self).__init__()
        self.module_cfg = module_cfg
        self.model_info_dict = model_info_dict

    def forward(self, batch_dict):
        raise NotImplementedError

    @property
    def output_feature_dims(self):
        raise NotImplementedError

    @property
    def output_feature_size(self):
        raise NotImplementedError
