#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/26 下午3:33
# @Author : PH
# @Version：V 0.1
# @File : SECOND.py
# @desc :
from .detect_model_base import Detect3DBase


class SECOND(Detect3DBase):

    def __init__(self, model_cfg, num_class, data_infos):
        super().__init__(model_cfg=model_cfg, num_class=num_class, data_infos=data_infos)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
