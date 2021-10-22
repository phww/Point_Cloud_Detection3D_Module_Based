#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/9 下午2:02
# @Author : PH
# @Version：V 0.1
# @File : bboxes.py
# @desc :
import torch


class BBoxes:

    # x y z h w l (r) or x y h w (r) + extra infos
    def __init__(self, bboxes, extra_infos=None, dims=2, with_rot=True, coder=None):
        self.num_bbox = bboxes.size(0)
        self.bboxex = bboxes
        self.xs = bboxes[:, 0]
        self.ys = bboxes[:, 1]
        self.zs = torch.zeros_like(self.xs)
        if dims == 3:
            self.zs = bboxes[:, dims]
        self.hs = bboxes[:, dims + 1]
        self.ws = bboxes[:, dims + 2]
        self.ls = None
        if dims == 3:
            self.ls = bboxes[:, dims + 3]
        self.rs = torch.zeros_like(self.xs)
        if with_rot:
            self.rs = bboxes[:, -1]
        self.coder = coder

    def encode(self):
        self.coder.encode()

    def decode(self):
        self.coder.decode()

    def calc_iou_with(self, bboxes):
        pass
