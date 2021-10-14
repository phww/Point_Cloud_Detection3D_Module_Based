#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 下午2:07
# @Author : PH
# @Version：V 0.1
# @File : __init__.py.py
# @desc :
from .anchor_gen_base import MultiClsAnchorGenerator, AnchorGenerator

__all__ = {
    "MultiClsAnchorGenerator": MultiClsAnchorGenerator,
    "AnchorGenerator"        : AnchorGenerator
}


def reshape_flatten_anchor_id(flatten_id, raw_anchor_shape):
    raw_anchor_shape = list(raw_anchor_shape) if not isinstance(raw_anchor_shape, list) else raw_anchor_shape
    raw_ids = []
    for cur_size in reversed(raw_anchor_shape):
        cur_id = flatten_id % cur_size
        flatten_id = int(flatten_id / cur_size)
        raw_ids.append(cur_id)
    ret = list(reversed(raw_ids))
    return ret
