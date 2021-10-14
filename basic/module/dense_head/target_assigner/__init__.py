#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 下午2:09
# @Author : PH
# @Version：V 0.1
# @File : __init__.py.py
# @desc :
from .maxiou_assigner import MaxIouTargetAssigner

__all__ = {
    "MaxIouTargetAssigner": MaxIouTargetAssigner,
}
