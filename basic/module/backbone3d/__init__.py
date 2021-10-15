#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 下午3:56
# @Author : PH
# @Version：V 0.1
# @File : __init__.py.py
# @desc :
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x

__all__ = {'VoxelBackBone8x'   : VoxelBackBone8x,
           'VoxelResBackBone8x': VoxelResBackBone8x
           }
