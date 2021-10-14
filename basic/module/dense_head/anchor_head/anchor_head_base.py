#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 下午2:11
# @Author : PH
# @Version：V 0.1
# @File : anchor_head_base.py
# @desc :
import torch
import torch.nn as nn
from .. import anchor_generator, target_assigner
from ..anchor_generator import reshape_flatten_anchor_id
from basic import loss


class AnchorHeadBase(nn.Module):

    def __init__(self, model_cfg, model_info_dict):
        super(AnchorHeadBase, self).__init__()
        self.module_cfg = model_cfg.MODEL.DENSE_HEAD
        self.model_info_dict = model_info_dict.copy()
        self.anchor_generator = self.build_anchor_generator()
        self.target_assigner = self.build_target_assigner()
        self.cls_loss_fn, self.reg_loss_fn = self.build_loss()
        self.cls_layer, self.reg_layer = self.init_input_layer()

    def forward(self, inputs, gts, gt_labels=None, **kwargs):
        # 1.get cls_pred and reg_pred map
        cls_pred = self.cls_layer(inputs)  # B,C*A,H,W
        reg_pred = self.reg_layer(inputs)  # B,7*A,H,W

        # 2.generate anchors based on inputs shape
        anchors = self.anchor_generator.gen_anchors(flatten_output=True)

        # 3.assign target for every anchor
        if gt_labels is None:
            gt_labels = gts[..., -1]  # gts:B,N,7+class
            gts = gts[..., :-1]
        target_dict, batch_bbox_ids_dict = self.target_assigner.assign(gts, anchors, gt_labels)

        # 4.calculate loss
        cls_pred_pos = self.pred_map_sampling(cls_pred,
                                              batch_bbox_ids_dict['pos'],
                                              )
        cls_pred_neg = self.pred_map_sampling(cls_pred,
                                              batch_bbox_ids_dict['neg'],
                                              )
        reg_pred = self.pred_map_sampling(reg_pred,
                                          batch_bbox_ids_dict['pos'],
                                          )
        cls_pred = torch.cat([cls_pred_neg, cls_pred_pos], dim=0)
        cls_loss = self.cls_loss_fn(cls_pred, target_dict['cls_labels'])
        reg_loss = self.reg_loss_fn(reg_pred, target_dict['reg_labels'])
        weights_dict = self.module_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        tol_loss = cls_loss * weights_dict['cls_weight'] + reg_loss * weights_dict['reg_weight']
        return tol_loss

    def predict_bbox(self, **kwargs):
        raise NotImplementedError

    def init_input_layer(self, **kwargs):
        input_channels = self.model_info_dict['cur_point_feature_dims']
        num_anchors_per_localization = self.anchor_generator.num_anchors_per_localization
        num_anchors_dims = self.anchor_generator.ndim
        num_class = len(self.model_info_dict['class_names']) + 1  # foreground + background
        cls_layer = nn.Conv2d(input_channels, num_class * num_anchors_per_localization, kernel_size=(1, 1))
        reg_layer = nn.Conv2d(input_channels, num_anchors_dims * num_anchors_per_localization, kernel_size=(1, 1))
        return cls_layer, reg_layer

    def build_anchor_generator(self, **kwargs):
        cfg = self.module_cfg.get("ANCHOR_GENERATOR_CONFIG", None)
        assert cfg is not None, "ANCHOR_GENERATOR_CONFIG Not Exist!"
        anchor_gen = anchor_generator.__all__[cfg.NAME](anchor_gen_cfg=cfg,
                                                        model_info_dict=self.model_info_dict
                                                        )
        self.model_info_dict['raw_anchor_shape'] = anchor_gen.shape
        return anchor_gen

    def build_target_assigner(self, **kwargs):
        cfg = self.module_cfg.get("TARGET_ASSIGNER_CONFIG", None)
        assert cfg is not None, "TARGET_ASSIGNER_CONFIG Not Exist!"
        target_assign = target_assigner.__all__[cfg.NAME](assigner_cfg=cfg,
                                                          model_info_dict=self.model_info_dict
                                                          )
        return target_assign

    def build_loss(self, **kwargs):
        cfg = self.module_cfg.get("LOSS_CONFIG", None)
        assert cfg is not None, "LOSS_CONFIG Not Exist!"
        cls_loss_fn = loss.__all__[cfg.CLS_LOSS.NAME](**cfg.CLS_LOSS)
        reg_loss_fn = loss.__all__[cfg.REG_LOSS.NAME](**cfg.REG_LOSS)
        return cls_loss_fn, reg_loss_fn

    def pred_map_sampling(self, pred_map, batch_bbox_ids):
        """
        Sample the perd map.Make sure only the sampled data and
        the corresponding pred are used to calculate loss
        Args:
            pred_map: tensor[B,C*A,(D),H,W]
            batch_bbox_ids: tensor M*(batch_id, bbox_id)

        Returns:
            sample_ret: tensor[N',C]
        """
        raw_anchor_shape = self.anchor_generator.shape
        num_anchors_per_loc = self.anchor_generator.num_anchors_per_localization
        batch_ids = batch_bbox_ids[:, 0]
        bbox_ids = batch_bbox_ids[:, 1]
        raw_bbox_ids = []
        for bbox_id in bbox_ids:
            raw_bbox_id = reshape_flatten_anchor_id(flatten_id=bbox_id.item(), raw_anchor_shape=raw_anchor_shape)
            raw_bbox_ids.append(raw_bbox_id)
        raw_bbox_ids = torch.tensor(raw_bbox_ids, dtype=torch.long, device=pred_map.device)
        view_shape = (pred_map.size(0), pred_map.size(1) // num_anchors_per_loc, *raw_anchor_shape)
        # B, C, A, H, W -> B, C, W, H, D(1), dim_size, dim_rot. A = dim_size * dim_rot
        pred_map = pred_map.view(view_shape)
        dimx, dimy, dimz, dim_size, dim_rot = [ids.squeeze(1) for ids in torch.split(raw_bbox_ids, 1, dim=1)]
        sample_ret = pred_map[batch_ids, :, dimx, dimy, dimz, dim_size, dim_rot]
        return sample_ret
