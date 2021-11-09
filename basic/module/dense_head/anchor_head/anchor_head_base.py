#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 下午2:11
# @Author : PH
# @Version：V 0.1
# @File : anchor_head_base.py
# @desc :
import torch
import torch.nn as nn
import einops
from .. import anchor_generator, target_assigner
from ..anchor_generator import reshape_flatten_anchor_id
from basic import loss
from basic.module.dense_head.anchor_generator import reshape_flatten_anchor_id


class AnchorHeadBase(nn.Module):

    def __init__(self, module_cfg, model_info_dict, **kwargs):
        super(AnchorHeadBase, self).__init__()
        self.in_channels = module_cfg.get('in_channels', None)
        self.module_cfg = module_cfg
        self.model_info_dict = model_info_dict
        self.anchor_generator = self.build_anchor_generator()
        self.target_assigner = self.build_target_assigner()
        self.cls_loss_fn, self.reg_loss_fn = self.build_loss()
        self.cls_layer, self.reg_layer = self.init_input_layer()

    def forward(self, batch_dict, **kwargs):
        raise NotImplementedError

    def train_assign(self, anchors, cls_pred, reg_pred, gts, gt_labels):
        raise NotImplementedError

    def predict_proposals(self, cls_pred, reg_pred, anchors):
        raise NotImplementedError

    def init_input_layer(self, **kwargs):
        input_channels = self.in_channels
        num_anchors_per_localization = self.anchor_generator.num_anchors_per_localization
        num_anchors_dims = self.anchor_generator.ndim
        if not self.model_info_dict.get('use_sigmoid', False):
            num_class = len(self.model_info_dict['class_names']) + 1  # foreground + background(0)
        else:
            num_class = 1
        cls_layer = nn.Conv2d(input_channels, num_class * num_anchors_per_localization, kernel_size=(1, 1))
        reg_layer = nn.Conv2d(input_channels, num_anchors_dims * num_anchors_per_localization, kernel_size=(1, 1))
        return cls_layer, reg_layer

    def calc_loss(self, cls_pred, reg_pred, assign_result):
        cls_loss = self.cls_loss_fn(inputs=cls_pred,
                                    targets=assign_result.cls_targets,
                                    weights=assign_result.cls_weights,
                                    avg_factor=self.model_info_dict['sample_size']
                                    )

        # if no positive bbox, reg_loss = 0
        reg_loss = 0.
        if assign_result.bbox_targets is not None:
            reg_loss = self.reg_loss_fn(inputs=reg_pred,
                                        targets=assign_result.bbox_targets,
                                        weights=assign_result.bbox_weights,
                                        avg_factor=self.model_info_dict['sample_size']
                                        )
        weights_dict = self.module_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        tol_loss = cls_loss * weights_dict['cls_weight'] + reg_loss * weights_dict['reg_weight']
        loss_dict = {
            "tol_loss": tol_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss
        }

        return loss_dict

    def build_anchor_generator(self, **kwargs):
        cfg = self.module_cfg.get("ANCHOR_GENERATOR_CONFIG", None)
        assert cfg is not None, "ANCHOR_GENERATOR_CONFIG Not Exist!"
        anchor_gen = anchor_generator.__all__[cfg.NAME](anchor_gen_cfg=cfg,
                                                        model_info_dict=self.model_info_dict
                                                        )
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
        if cfg.CLS_LOSS.NAME == 'CrossEntropyLoss':
            assert not cfg.CLS_LOSS.get('use_sigmoid', False)
            cls_loss_fn = nn.CrossEntropyLoss()
        else:
            cls_loss_fn = loss.__all__[cfg.CLS_LOSS.NAME](**cfg.CLS_LOSS)
        reg_loss_fn = loss.__all__[cfg.REG_LOSS.NAME](**cfg.REG_LOSS)
        self.model_info_dict['use_sigmoid'] = cfg.CLS_LOSS.get('use_sigmoid', False)
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
        num_anchors_per_loc = self.anchor_generator.num_anchors_per_localization
        B = pred_map.size(0)
        C = pred_map.size(1) // num_anchors_per_loc
        pred_map = pred_map.permute(0, 2, 3, 1).reshape(B, -1, C)
        raw_anchor_shape = self.anchor_generator.shape

        batch_ids = batch_bbox_ids[:, 0]
        bbox_ids = batch_bbox_ids[:, 1]
        # raw_bbox_ids = []
        # for bbox_id in bbox_ids:
        #     raw_bbox_id = reshape_flatten_anchor_id(flatten_id=bbox_id.item(), raw_anchor_shape=raw_anchor_shape)
        #     raw_bbox_ids.append(raw_bbox_id)
        # raw_bbox_ids = torch.tensor(raw_bbox_ids, dtype=torch.long, device=pred_map.device)
        # view_shape = (pred_map.size(0), pred_map.size(1) // num_anchors_per_loc, *raw_anchor_shape)
        # # B, C, A, H, W -> B, C, W, H, D(1), dim_size, dim_rot. A = dim_size * dim_rot
        # pred_map = pred_map.view(view_shape).contiguous()
        # dimx, dimy, dimz, dim_size, dim_rot = [ids.squeeze(1) for ids in torch.split(raw_bbox_ids, 1, dim=1)]
        # sample_ret = pred_map[batch_ids, :, dimx, dimy, dimz, dim_size, dim_rot]
        sample_ret = pred_map[batch_ids, bbox_ids]
        return sample_ret
