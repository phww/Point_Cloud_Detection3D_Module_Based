#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/2 下午1:51
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
            return output_dict
        else:
            # 3.during predicting, figure out the foregrounds
            # output_dict = self.predict_bbox(cls_pred, reg_pred, anchors)
            proposals = self.predict_proposals(cls_pred, reg_pred, anchors)
            return proposals

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
        # ce = nn.CrossEntropyLoss()
        # cls_loss = ce(cls_pred, target_dict['cls_labels'].long())
        cls_loss = self.cls_loss_fn(cls_pred, target_dict['cls_labels'])
        # if no positive bbox, reg_loss = 0
        reg_loss = 0.
        if reg_pred is not None:
            reg_loss = self.reg_loss_fn(reg_pred, target_dict['reg_labels'])
        weights_dict = self.module_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        tol_loss = cls_loss * weights_dict['cls_weight'] + reg_loss * weights_dict['reg_weight']
        loss_dict = {
            "tol_loss": tol_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss
        }
        return loss_dict

    def train_assign(self, anchors, cls_pred, reg_pred, gts, gt_labels):
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
            'assign_result': assign_result
        }
        return output_dict

    def _predict_all_bboxes(self, cls_pred, reg_pred, anchors):
        """
        predict scores and bboxes for all anchors
        Args:
            cls_pred: [torch.tensor] B,C*A,H,W
            reg_pred: [torch.tensor] B,7*A,H,W
            anchors:  [torch.tensor] num_anchors, 7. num_anchors = A*H*W

        Returns:
            pred_scores: [torch.tensor] B, num_anchors, C
            pred_bbox: [torch.tensor] B, num_anchors, 7

        """
        B = cls_pred.size(0)
        num_anchors = anchors.size(0)
        # B, C*A, H, W -> B, H, W, C*A -> B, num_anchors, C
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(B, num_anchors, -1)
        reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, num_anchors, -1)
        batch_anchors = anchors.unsqueeze(dim=0).repeat(B, 1, 1)  # B, num_anchors, 7
        pred_bboxes = self.target_assigner.bbox_encoder.decode(reg_pred, batch_anchors)
        # pred_bboxes = batch_anchors
        pred_scores = torch.softmax(cls_pred, dim=-1)
        return pred_scores, pred_bboxes

    def predict_proposals_one_frame(self, scores, bboxes, k):
        assert 0 < k < scores.size(0)
        max_scores, _ = scores[:, 1:].max(dim=-1)
        _, topk_inds = max_scores.topk(k)
        topk_bboxes = bboxes[topk_inds]
        topk_scores = scores[topk_inds]
        return topk_scores, topk_bboxes

    def predict_proposals(self, cls_pred, reg_pred, anchors):
        B = cls_pred.size(0)
        pred_scores, pred_bboxes = self._predict_all_bboxes(cls_pred, reg_pred, anchors)
        proposals = []
        for i in range(B):
            frame_proposals = self.predict_proposals_one_frame(scores=pred_scores[i],
                                                               bboxes=pred_bboxes[i],
                                                               k=100
                                                               )
            proposals.append(frame_proposals)
        return proposals

    def predict_one_frame(self, frame_cls_scores, frame_reg, frame_topk_inds, frame_topk_labels, anchors):
        """

        Args:
            frame_cls_scores: N,C
            frame_reg: N,7
            frame_topk_inds: k
            anchors: A*H*W,7
            frame_topk_labels: k
        Returns:

        """
        pos_mask = frame_topk_labels > 0  # num_pos
        pos_bbox_inds = frame_topk_inds[pos_mask]

        anchor_ids = [reshape_flatten_anchor_id(pos_bbox_ind,
                                                self.model_info_dict['raw_anchor_shape']
                                                )
                      for pos_bbox_ind in pos_bbox_inds]
        print(anchor_ids)
        pos_labels = frame_topk_labels[pos_mask]
        frame_bbox = anchors[pos_bbox_inds]  # k, 7
        # frame_anchors = anchors[pos_bbox_inds]  # k, 7
        # frame_reg_pos = frame_reg[pos_bbox_inds]  # K, 7
        # frame_bbox = self.target_assigner.bbox_encoder.decode(frame_reg_pos, frame_anchors)  # k, 7
        frame_topk_cls_scores = frame_cls_scores[pos_bbox_inds]  # k, C

        return frame_bbox, pos_labels, frame_topk_cls_scores

    def predict_bbox(self, cls_pred, reg_pred, anchors, **kwargs):
        """

        Args:
            cls_pred: B,C*A,H,W
            reg_pred: B,7*A,H,W
            anchors: W*H*dimz(1)*num_size*num_rot,7. and num_size*num_rot = A
            **kwargs:

        Returns:

        """
        k = 100
        B, _, H, W = cls_pred.shape
        A = self.anchor_generator.num_anchors_per_localization
        num_anchors = anchors.size(0)
        # cls_pred = einops.rearrange(cls_pred, "B CA H W -> B (W H A) C", A=A)
        # reg_pred = einops.rearrange(reg_pred,  "B CA H W -> B (W H A) C", A=A)
        cls_pred = cls_pred.view(B, -1, A, H, W)  # B, C, A, H, W
        reg_pred = reg_pred.view(B, -1, A, H, W)  # B, 7, A, H, W
        # B, C, A, H, W -> B, W, H, A, C -> B, H*W*A, C
        cls_pred = cls_pred.permute(0, 4, 3, 2, 1).contiguous().view(B, W * H * A, -1)
        reg_pred = reg_pred.permute(0, 4, 3, 2, 1).contiguous().view(B, W * H * A, -1)
        # cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(B, num_anchors, -1)  # B, C, A, H, W
        # reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, num_anchors, -1)  # B, 7, A, H, W
        # batch_anchors = anchors.unsqueeze(dim=0).repeat(B, 1, 1)  # B, A*H*W, 7
        # decode_bbox = self.target_assigner.bbox_encoder.decode(reg_pred, batch_anchors)
        # B, C, A, H, W -> B, W, H, A, C -> B, H*W*A, C
        # cls_pred = cls_pred.permute(0, 3, 2, 4, 1).contiguous().view(B, -1, 2)
        # reg_pred = reg_pred.permute(0, 3, 2, 4, 1).contiguous().view(B, -1, 7)
        cls_scores = torch.softmax(cls_pred, dim=-1)  # B, N, C
        max_scores, argmax_scores = cls_scores.max(dim=-1)  # B, N
        _, batch_topk_inds = max_scores.topk(k, dim=-1)  # B, k
        # topk labels for every frame: B, k

        topk_labels = torch.stack([argmax_scores[i, batch_topk_inds[i]] for i in range(B)], dim=0)
        # topk_bbox = torch.stack([decode_bbox[i][batch_topk_inds[i]] for i in range(2)])
        # pos_mask = torch.where(topk_labels > 0)
        # pos_bbox = topk_bbox[pos_mask]
        # pos_labels = topk_labels[pos_mask]
        pred_bbox = []
        pred_bbox_labels = []
        frame_inds = []
        for i in range(B):
            frame_bbox, frame_bbox_labels, frame_scores = self.predict_one_frame(frame_cls_scores=cls_scores[i],
                                                                                 frame_reg=reg_pred[i],
                                                                                 frame_topk_inds=batch_topk_inds[i],
                                                                                 frame_topk_labels=topk_labels[i],
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