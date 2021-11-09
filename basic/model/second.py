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

    def __init__(self, top_cfg, data_infos):
        super(SECOND, self).__init__(top_cfg, data_infos)
        self.module_list = self.build_model()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss_dict = self.get_training_loss(batch_dict)
            return loss_dict
        else:
            # pred_bbox = batch_dict['pred_bbox']
            # pred_bbox_labels = batch_dict['pred_bbox_labels']
            # frame_inds = batch_dict['frame_inds']
            # self.post_processing(batch_dict)

            return batch_dict

    def get_training_loss(self, batch_dict):
        loss_dict = self.dense_head.calc_loss(cls_pred=batch_dict['cls_pred'],
                                              reg_pred=batch_dict['reg_pred'],
                                              assign_result=batch_dict['assign_result']
                                              )
        return loss_dict
