#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/9 下午2:57
# @Author : PH
# @Version：V 0.1
# @File : iou3d_calc.py
# @desc :
import torch

from basic.ops.pc_3rd_ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


class Iou3DCalculator:

    def __init__(self, **kwargs):
        self.calculator = boxes_iou3d_gpu

    def __call__(self, gts, bboxes, **kwargs):
        """

        Args:
            gts: B,N,7 or N,7
            bboxes: M ,7
            **kwargs:

        Returns:

        """

        ious = self.calculator(gts, bboxes)
        return ious
