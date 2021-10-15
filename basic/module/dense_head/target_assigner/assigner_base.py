#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/8 上午9:42
# @Author : PH
# @Version：V 0.1
# @File : assigner_base.py
# @desc :
from basic.utils.bbox import bbox_sampler, bbox_encoder, iou_calculator


class AssignerBase:

    def __init__(self, assigner_cfg, model_info_dict,**kwargs):
        self.module_cfg = assigner_cfg
        self.model_info_dict = model_info_dict
        self.device = assigner_cfg.DEVICE
        self.iou_calculator = self.build_iou_calculator()
        self.bbox_encoder = self.build_box_encoder()
        self.sampler = self.build_sampler()

    def assign(self, gts, bboxes, gt_labels, **kwargs):
        raise NotImplementedError

    def build_iou_calculator(self):
        cfg = self.module_cfg.get("IOU_CALCULATOR", None)
        assert cfg is not None, "IOU_CALCULATOR CONFIG Not Exist!"
        iou_calc = iou_calculator.__all__[cfg.NAME](**cfg)
        return iou_calc

    def build_box_encoder(self):
        cfg = self.module_cfg.get("BOX_ENCODER", None)
        if cfg is None:
            return None
        encoder = bbox_encoder.__all__[cfg.NAME](**cfg)
        return encoder

    def build_sampler(self):
        cfg = self.module_cfg.get("SAMPLER", None)
        if cfg is None:
            return None
        sampler = bbox_sampler.__all__[cfg.NAME](**cfg)
        return sampler
