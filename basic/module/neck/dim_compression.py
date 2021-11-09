#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/26 下午4:07
# @Author : PH
# @Version：V 0.1
# @File : dim_compression.py
# @desc :
from .neck_base import NECK
import einops


class DimCompression(NECK):

    def __init__(self, model_info_dict, dim=2, **kwargs):
        super(DimCompression, self).__init__(model_info_dict)
        self.dim = dim

    # @property
    # def output_feature_dims(self):
    #     return self._output_feature_dims

    # @property
    # def output_feature_size(self):
    #     size = self.model_info_dict['feature_map_size']
    #     size[self.dim - 2] = 1
    #     return size

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        assert self.dim > 1
        spconv_tensor = batch_dict['sp_feat3d']
        spatial_features = spconv_tensor.dense()
        if self.dim == 2:
            spatial_features = einops.rearrange(spatial_features, "B C D H W->B (C D) H W")
        if self.dim == 3:
            spatial_features = einops.rearrange(spatial_features, "B C D H W->B (C H) D W")
        if self.dim == 4:
            spatial_features = einops.rearrange(spatial_features, "B C D H W->B (C W) D H")
        batch_dict['dense_feat2d'] = spatial_features
        return batch_dict
