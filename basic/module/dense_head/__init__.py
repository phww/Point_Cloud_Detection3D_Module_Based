#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 下午3:57
# @Author : PH
# @Version：V 0.1
# @File : __init__.py.py
# @desc :
from .anchor_head.anchor3d_head import Anchor3DHead

__all__ = {
    'Anchor3DHead': Anchor3DHead
}
