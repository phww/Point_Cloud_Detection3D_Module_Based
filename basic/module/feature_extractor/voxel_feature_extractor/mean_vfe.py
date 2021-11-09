#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 下午4:32
# @Author : PH
# @Version：V 0.1
# @File : mean_vfe.py
# @desc :
import torch
from .vfe_base import VFEBase


class MeanVFE(VFEBase):

    def __init__(self, model_info_dict, is_normalize=True, **kwargs):
        super().__init__(model_info_dict=model_info_dict)
        self.is_normalize = is_normalize

    @property
    def output_feature_dims(self):
        return self.model_info_dict['cur_point_feature_dims']

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        if self.is_normalize:
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
