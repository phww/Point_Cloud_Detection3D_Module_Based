#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/9 下午2:52
# @Author : PH
# @Version：V 0.1
# @File : anchor3d_head.py
# @desc :
import torch

from .anchor_head_base import AnchorHeadBase


class Anchor3DHead(AnchorHeadBase):

    def __init__(self, top_cfg, model_info_dict):
        super(Anchor3DHead, self).__init__(module_cfg=top_cfg.MODEL.DENSE_HEAD, model_info_dict=model_info_dict)
        self.top_cfg = top_cfg

    def forward(self, batch_dict, **kwargs):
        inputs = batch_dict['dense_feat2d']
        gts = batch_dict['gt_boxes']
        gt_labels = batch_dict.get('gt_labels', None)
        # 1.get cls_pred and reg_pred map
        cls_pred = self.cls_layer(inputs)  # B,C*A,H,W
        reg_pred = self.reg_layer(inputs)  # B,7*A,H,W
        feat_map_size = list(cls_pred.shape[2:])
        if len(feat_map_size) == 2:
            feat_map_size.insert(0, 1)  # 1, H, W
        self.model_info_dict['feat_map_size'] = feat_map_size

        # 2.generate anchors based on inputs shape
        anchors = self.anchor_generator.gen_anchors(flatten_output=True, feature_map_size=feat_map_size)

        if self.training:
            # 3.during training, assign target for sampled anchor
            output_dict = self.train_assign(anchors, cls_pred, reg_pred, gts, gt_labels)
            return output_dict
        else:
            # 3.during predicting, figure out the proposals
            proposals = self.predict_proposals(cls_pred, reg_pred, anchors)
            batch_dict['proposal_dict'] = proposals
        return batch_dict

    def train_assign(self, anchors, cls_pred, reg_pred, gts, gt_labels):
        if self.model_info_dict['use_sigmoid']:
            num_class = 1
        else:
            num_class = len(self.model_info_dict['class_names']) + 1
        bbox_dim = self.anchor_generator.ndim
        B = cls_pred.size(0)
        cls_pred = cls_pred.permute(0, 3, 2, 1).reshape(B, -1, num_class)
        reg_pred = reg_pred.permute(0, 3, 2, 1).reshape(B, -1, bbox_dim)
        if gt_labels is None:
            gt_labels = gts[..., -1]  # gts:B,N,7+class
            gts = gts[..., :-1]
        assign_result = self.target_assigner.assign(gts, anchors, gt_labels)
        # assign_result.add_gts(gts, gt_labels)
        output_dict = {
            'cls_pred'     : cls_pred,
            'reg_pred'     : reg_pred,
            'assign_result': assign_result
        }
        return output_dict

    def predict_proposals(self, cls_pred, reg_pred, anchors):
        B = cls_pred.size(0)
        pred_scores, pred_bboxes = self._predict_all_bboxes(cls_pred, reg_pred, anchors)
        # proposals = []
        # proposal_scores = []
        # proposal_labels = []
        # for i in range(B):
        #     frame_topk_scores, frame_topk_bboxes, frame_topk_labels = self.predict_proposals_one_frame(
        #             scores=pred_scores[i],
        #             bboxes=pred_bboxes[i],
        #             k=self.top_cfg.INFERENCE_CONFIG.num_topk
        #     )
        #     proposals.append(frame_topk_bboxes)
        #     proposal_scores.append(frame_topk_scores)
        #     proposal_labels.append(frame_topk_labels)
        # proposals = torch.stack(proposals, dim=0)
        # proposal_scores = torch.stack(proposal_scores, dim=0)
        # proposal_labels = torch.stack(proposal_labels, dim=0)
        proposal_dict = {
            'proposals'      : pred_bboxes,
            'proposal_scores': pred_scores,
            # 'proposal_labels': proposal_labels
        }
        return proposal_dict

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
        cls_pred = cls_pred.permute(0, 3, 2, 1).contiguous().view(B, num_anchors, -1)
        reg_pred = reg_pred.permute(0, 3, 2, 1).contiguous().view(B, num_anchors, -1)
        batch_anchors = anchors.unsqueeze(dim=0).repeat(B, 1, 1)  # B, num_anchors, 7
        pred_bboxes = self.target_assigner.bbox_encoder.decode(reg_pred, batch_anchors)
        # pred_bboxes = batch_anchors
        if self.model_info_dict['use_sigmoid']:
            pred_scores = torch.sigmoid(cls_pred.unsqueeze(dim=-1))
        else:
            pred_scores = torch.softmax(cls_pred, dim=-1)
        return pred_scores, pred_bboxes

    def predict_proposals_one_frame(self, scores, bboxes, k):
        assert 0 < k < scores.size(0)
        if self.model_info_dict['use_sigmoid']:
            max_scores, label = scores.max()
        else:
            max_scores, label = scores[:, 1:].max(dim=-1)
        _, topk_inds = max_scores.topk(k)
        topk_bboxes = bboxes  # [topk_inds]
        topk_scores = scores  # [topk_inds]
        topk_labels = label[topk_inds] + 1
        return topk_scores, topk_bboxes, topk_labels
