#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 下午2:11
# @Author : PH
# @Version：V 0.1
# @File : anchor_head_base.py
# @desc :
class AnchorHeadBase:

    def __init__(self):
        pass

    def forward(self, **kwargs):
        raise NotImplementedError

    def predict_bbox(self, **kwargs):
        raise NotImplementedError

    def build_input_layer(self, **kwargs):
        pass

    def build_anchor_generator(self, **kwargs):
        pass

    def build_anchor_encoder(self, **kwargs):
        pass

    def build_target_assigner(self, **kwargs):
        pass

    def build_loss(self, **kwargs):
        pass
