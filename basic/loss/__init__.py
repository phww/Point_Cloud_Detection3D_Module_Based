#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/8 下午6:32
# @Author : PH
# @Version：V 0.1
# @File : __init__.py.py
# @desc :
from .focal_loss import FocalLoss
from .smoothL1_loss import SmoothL1Loss

__all__ = {
    "FocalLoss"   : FocalLoss,
    "SmoothL1Loss": SmoothL1Loss
}
