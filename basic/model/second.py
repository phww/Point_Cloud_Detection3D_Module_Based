#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/26 下午3:33
# @Author : PH
# @Version：V 0.1
# @File : second.py
# @desc :
from .detect_model_base import Detect3DBase


class SECOND(Detect3DBase):

    def __init__(self, model_cfg, data_infos):
        super(SECOND, self).__init__(model_cfg, data_infos)
        self.module_list = self.build_model()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss= self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_rpn= self.dense_head.calc_loss(batch_dict)
        # tb_dict = {
        #     'loss_rpn': loss_rpn.item(),
        #     **tb_dict
        # }

        loss = loss_rpn
        return loss
