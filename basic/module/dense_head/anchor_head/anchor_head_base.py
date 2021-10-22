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

    def __init__(self, module_cfg, model_info_dict):
        super(AnchorHeadBase, self).__init__()
        self.module_cfg = module_cfg
        self.model_info_dict = model_info_dict.copy()
        self.anchor_generator = self.build_anchor_generator()
        self.target_assigner = self.build_target_assigner()
        self.cls_loss_fn, self.reg_loss_fn = self.build_loss()
        self.cls_layer, self.reg_layer = self.init_input_layer()

    def forward(self, batch_dict, **kwargs):
        inputs = batch_dict['spatial_features_2d']
        gts = batch_dict['gt_boxes']
        gt_labels = batch_dict.get('gt_labels', None)
        # 1.get cls_pred and reg_pred map
        cls_pred = self.cls_layer(inputs)  # B,C*A,H,W
        reg_pred = self.reg_layer(inputs)  # B,7*A,H,W

        # 2.generate anchors based on inputs shape
        anchors = self.anchor_generator.gen_anchors(flatten_output=True)
        self.model_info_dict['anchors'] = anchors
        if self.training:
            # 3.during training, assign target for sampled anchor
            output_dict = self.train_assign(anchors, cls_pred, reg_pred, gt_labels, gts)
        else:
            # 3.during predicting, figure out the foregrounds
            output_dict = self.predict_bbox(cls_pred, reg_pred, anchors)
        return output_dict

    def init_input_layer(self, **kwargs):
        input_channels = self.model_info_dict['cur_point_feature_dims']
        num_anchors_per_localization = self.anchor_generator.num_anchors_per_localization
        num_anchors_dims = self.anchor_generator.ndim
        num_class = len(self.model_info_dict['class_names']) + 1  # foreground + background(0)
        cls_layer = nn.Conv2d(input_channels, num_class * num_anchors_per_localization, kernel_size=(1, 1))
        reg_layer = nn.Conv2d(input_channels, num_anchors_dims * num_anchors_per_localization, kernel_size=(1, 1))
        return cls_layer, reg_layer

    def calc_loss(self, output_dict):
        cls_pred = output_dict['cls_pred']
        reg_pred = output_dict['reg_pred']
        target_dict = output_dict['target_dict']
        cls_loss = self.cls_loss_fn(cls_pred, target_dict['cls_labels'])
        # if no positive bbox, reg_loss = 0
        reg_loss = 0.
        if reg_pred is not None:
            reg_loss = self.reg_loss_fn(reg_pred, target_dict['reg_labels'])
        weights_dict = self.module_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        tol_loss = cls_loss * weights_dict['cls_weight'] + reg_loss * weights_dict['reg_weight']
        return tol_loss

    def train_assign(self, anchors, cls_pred, reg_pred, gt_labels, gts):
        if gt_labels is None:
            gt_labels = gts[..., -1]  # gts:B,N,7+class
            gts = gts[..., :-1]
        target_dict, batch_bbox_ids_dict = self.target_assigner.assign(gts, anchors, gt_labels)
        self.model_info_dict['batch_bbox_ids_dict'] = batch_bbox_ids_dict
        cls_pred_neg = None
        cls_pred_pos = None
        sampled_reg_pred = None
        if batch_bbox_ids_dict['pos'] is not None:
            cls_pred_pos = self.pred_map_sampling(cls_pred,
                                                  batch_bbox_ids_dict['pos']
                                                  )
        if batch_bbox_ids_dict['neg'] is not None:
            cls_pred_neg = self.pred_map_sampling(cls_pred,
                                                  batch_bbox_ids_dict['neg'],
                                                  )
        if batch_bbox_ids_dict['pos'] is not None:
            sampled_reg_pred = self.pred_map_sampling(reg_pred,
                                                      batch_bbox_ids_dict['pos'],
                                                      )
        sampled_cls_pred = torch.cat([cls_pred_neg, cls_pred_pos], dim=0) if cls_pred_pos is not None else cls_pred_neg
        output_dict = {
            'cls_pred'   : sampled_cls_pred,
            'reg_pred'   : sampled_reg_pred,
            'target_dict': target_dict
        }
        return output_dict

    def predict_one_frame(self, frame_cls, frame_reg, frame_pos_mask, anchors):
        """

        Args:
            frame_cls: N,C
            frame_reg: N,7
            frame_pos_mask: N
            anchors: A*H*W,7

        Returns:

        """
        frame_anchors = anchors[frame_pos_mask]
        frame_reg_pos = frame_reg[frame_pos_mask]
        frame_bbox = self.target_assigner.bbox_encoder.decode(frame_reg_pos, frame_anchors)
        frame_cls_pos = frame_cls[frame_pos_mask]
        return frame_bbox, frame_cls_pos

    def predict_bbox(self, cls_pred, reg_pred, anchors, **kwargs):
        """

        Args:
            cls_pred: B,C*A,H,W
            reg_pred: B,7*A,H,W
            anchors: W*H*dimz(1)*num_size*num_rot,7. and num_size*num_rot = A
            **kwargs:

        Returns:

        """
        B, _, H, W = cls_pred.shape
        A = self.anchor_generator.num_anchors_per_localization
        cls_pred = cls_pred.view(B, cls_pred.size(1) // A, H, W, -1).contiguous()  # B, C, H, W, A
        reg_pred = reg_pred.view(B, reg_pred.size(1) // A, H, W, -1).contiguous()  # B, 7, H, W, A
        # B, C, H, W, A -> B, H, W, A, C -> B, H*W*A, C
        cls_pred = cls_pred.permute(0, 2, 3, 4, 1).view(B, H * W * A, -1).contiguous()
        reg_pred = reg_pred.permute(0, 2, 3, 4, 1).view(B, H * W * A, -1).contiguous()
        pred_cls_ids = torch.softmax(cls_pred, dim=-1).argmax(dim=-1)
        pos_mask = pred_cls_ids > 0  # B, N
        pred_bbox = []
        pred_bbox_labels = []
        frame_inds = []
        for i in range(B):
            frame_bbox, frame_bbox_labels = self.predict_one_frame(frame_cls=pred_cls_ids[i],
                                                                   frame_reg=reg_pred[i],
                                                                   frame_pos_mask=pos_mask[i],
                                                                   anchors=anchors
                                                                   )
            if frame_bbox.size(0) > 0:
                pred_bbox.extend(frame_bbox.tolist())
                pred_bbox_labels.extend(frame_bbox_labels.tolist())
                frame_inds.extend([i] * frame_bbox.size(0))
        pred_bbox = torch.tensor(pred_bbox, dtype=torch.float) if len(pred_bbox) > 0 else None
        pred_bbox_labels = torch.tensor(pred_bbox_labels, dtype=torch.long) if len(pred_bbox_labels) > 0 else None
        frame_inds = torch.tensor(frame_inds, dtype=torch.long) if len(frame_inds) > 0 else None

        output_dict = {
            'pred_bbox'       : pred_bbox,
            'pred_bbox_labels': pred_bbox_labels,
            'frame_inds'      : frame_inds
        }
        return output_dict

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
        pred_map = pred_map.view(view_shape).contiguous()
        dimx, dimy, dimz, dim_size, dim_rot = [ids.squeeze(1) for ids in torch.split(raw_bbox_ids, 1, dim=1)]
        sample_ret = pred_map[batch_ids, :, dimx, dimy, dimz, dim_size, dim_rot]
        return sample_ret
