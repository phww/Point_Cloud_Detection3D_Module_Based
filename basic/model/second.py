#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/26 下午3:33
# @Author : PH
# @Version：V 0.1
# @File : second.py
# @desc :
import torch
# from basic.utils.nms_utils import class_agnostic_nms
from .detect_model_base import Detect3DBase
from ..ops.pc_3rd_ops.roiaware_pool3d import roiaware_pool3d_utils


class SECOND(Detect3DBase):

    def __init__(self, top_cfg):
        super(SECOND, self).__init__(top_cfg)
        self.module_list = self.build_model()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss_dict = self.get_training_loss(batch_dict)
            return loss_dict
        else:
            pred_output = batch_dict
            if self.top_cfg.INFERENCE_CONFIG.get('POST_PROCESSING', None) is not None:
                frame_dict_list = self.post_processing(batch_dict=batch_dict,
                                                       gts=batch_dict['gt_boxes'][..., :-1],
                                                       gt_labels=batch_dict['gt_boxes'][..., -1],
                                                       post_cfg=self.top_cfg.INFERENCE_CONFIG.POST_PROCESSING
                                                       )
                pred_output = frame_dict_list
            return pred_output

    def get_training_loss(self, batch_dict):
        loss_dict = self.dense_head.calc_loss(cls_pred=batch_dict['cls_pred'],
                                              reg_pred=batch_dict['reg_pred'],
                                              assign_result=batch_dict['assign_result']
                                              )
        return loss_dict
