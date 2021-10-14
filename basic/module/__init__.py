#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 上午11:08
# @Author : PH
# @Version：V 0.1
# @File : __init__.py.py
# @desc :
from .feature_extractor.voxel_feature_extractor.mean_vfe import MeanVFE
from .backbone3d.spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x

alternative_module_list = [
    'feature_extractor',
    'backbone3d',
    'neck',
    'backbone2d',
    'dense_head',
    'point_head',
    'roi_head'
]

