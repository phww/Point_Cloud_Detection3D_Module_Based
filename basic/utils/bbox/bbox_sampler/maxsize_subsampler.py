#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/12 下午3:59
# @Author : PH
# @Version：V 0.1
# @File : maxsize_subsampler.py
# @desc :
import torch


class MaxSizeSubSampler:

    def __init__(self, sample_size=50, pos_ratio=None, **kwargs):
        self.max_size = sample_size
        self.pos_ratio = pos_ratio

    def __call__(self, pos_tuples, neg_tuples, **kwargs):
        num_pos = pos_tuples.shape[0] if pos_tuples is not None else 0
        num_neg = neg_tuples.shape[0] if neg_tuples is not None else 0
        if self.pos_ratio is not None:
            num_select_pos = min(int(self.max_size * self.pos_ratio), num_pos)
        else:
            num_select_pos = min(num_pos, self.max_size)
        num_select_neg = min(self.max_size - num_select_pos, num_neg)
        if num_select_pos != 0:
            pos_mask = torch.randperm(num_pos, device=pos_tuples.device)[:num_select_pos]
            select_pos = pos_tuples[pos_mask]
        else:
            select_pos = None
        if num_select_neg != 0:
            neg_mask = torch.randperm(num_neg, device=neg_tuples.device)[:num_select_neg]
            select_neg = neg_tuples[neg_mask]
        else:
            select_neg = None
        return select_pos, select_neg
