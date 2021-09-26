#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/26 下午4:07
# @Author : PH
# @Version：V 0.1
# @File : normal_compression.py
# @desc :
import torch.nn as nn
import einops


class DimCompression(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict, dim=3):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        assert dim > 2
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        if dim == 3:
            spatial_features = einops.rearrange(spatial_features, "B C D H W->B (C D) H W")
        if dim == 4:
            spatial_features = einops.rearrange(spatial_features, "B C D H W->B (C H) D W")
        if dim == 5:
            spatial_features = einops.rearrange(spatial_features, "B C D H W->B (C W) D H")
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
