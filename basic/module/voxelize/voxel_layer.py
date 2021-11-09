#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/5 下午1:35
# @Author : PH
# @Version：V 0.1
# @File : voxel_layer.py
# @desc :
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from basic.utils.common_utils import put_data_to_gpu


class VoxelLayer(nn.Module):

    def __init__(self, model_info_dict,
                 point_cloud_range,
                 voxel_size=(0.05, 0.05, 0.1),
                 max_points_pre_voxel=5,
                 max_voxels=dict(train=16000, test=40000),
                 full_mean=False,
                 use_lead_xyz=True,
                 **kwargs,
                 ):
        super(VoxelLayer, self).__init__()
        self.model_info_dict = model_info_dict
        self.point_cloud_range = np.array(point_cloud_range) if isinstance(point_cloud_range, list) else point_cloud_range
        self.max_points = max_points_pre_voxel
        self.max_voxel = max_voxels
        self.full_mean = full_mean
        self.use_lead_xyz = use_lead_xyz
        self.voxel_size = voxel_size
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.voxel_generator = None
        self.mode = 'train' if self.training else 'test'

        self.model_info_dict['grid_size'] = self.grid_size
        self.model_info_dict['voxel_size'] = voxel_size

    def forward(self, data_dict, voxel_generator=None, keep_points=True):
        if voxel_generator is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            self.voxel_generator = VoxelGenerator(
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.model_info_dict['point_cloud_range'],
                    max_num_points=self.max_points,
                    max_voxels=self.max_voxel[self.mode],
                    full_mean=self.full_mean
            )


        else:
            self.voxel_generator = voxel_generator

        voxels = []
        coordinates = []
        num_points = []
        points = data_dict['points'][:, 1:]
        points_batch_inds = data_dict['points'][:, 0]
        for b in range(data_dict['batch_size']):
            frame_points = points[points_batch_inds == b]

            if type(frame_points) == torch.Tensor:
                frame_points = frame_points.cpu().numpy()
            voxel_output = self.voxel_generator.generate(frame_points)
            if isinstance(voxel_output, dict):
                frame_voxels, frame_coordinates, frame_num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                frame_voxels, frame_coordinates, frame_num_points = voxel_output

            if not getattr(self, 'use_lead_xyz', True):
                frame_voxels = frame_voxels[..., 3:]  # remove xyz in voxels(N, 3)
            voxels.append(frame_voxels)
            num_points.append(frame_num_points)
            coor_pad = np.pad(frame_coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=b)
            coordinates.append(coor_pad)

        data_dict['voxels'] = np.concatenate(voxels, axis=0)
        data_dict['voxel_coords'] = np.concatenate(coordinates, axis=0)
        data_dict['voxel_num_points'] = np.concatenate(num_points, axis=0)

        if not keep_points:
            del data_dict['points']
        data_dict = put_data_to_gpu(data_dict)
        return data_dict

    def voxelize_one_frame(self, points):
        pass
