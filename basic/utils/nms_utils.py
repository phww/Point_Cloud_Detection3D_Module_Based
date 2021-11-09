#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/4 下午2:11
# @Author : PH
# @Version：V 0.1
# @File : nms_utils.py
# @desc :
import torch
from numpy import indices

from ..ops.pc_3rd_ops.iou3d_nms import iou3d_nms_utils


class NMS3D:

    def __init__(self, nms_cfg, scores_thresh=None):
        self.nms_cfg = nms_cfg
        self.scores_thr = scores_thresh

    def single_class_nms(self, cls_scores, bbox_preds):
        """

        Args:
            cls_scores: B, N
            bbox_preds: B, N, 7 + C

        Returns:

        """
        nms_result = []
        for i, (bbox_pred, cls_score) in enumerate(zip(bbox_preds, cls_scores)):
            frame_bbox_pred, frame_cls_score = self._single_class_nms_for_frame(cls_score, bbox_pred)
            frame_id = torch.ones((frame_cls_score.size(0), 1), dtype=torch.float, device=frame_bbox_pred.device) * i
            frame_result = torch.cat([frame_id, frame_bbox_pred, frame_cls_score.unsqueeze(dim=1)], dim=1)
            nms_result.append(frame_result)
        nms_result = torch.cat(nms_result, dim=0)
        return nms_result

    def multi_classes_nms(self, bbox_scores, bbox_preds):
        """
        Args:
            bbox_scores: (B, N, num_class)
            box_preds: (B, N, 7 + C)

        Returns:

        """
        B, N, C = bbox_scores.shape[:]
        pred_scores, pred_labels, pred_boxes = [], [], []
        for c in range(1, C):
            cls_scores = bbox_scores[:, c]  # B, N

            selected = []
            selected_cls_bbox, selected_cls_scores = self.single_class_nms(cls_scores=cls_scores,
                                                                           bbox_preds=bbox_preds
                                                                           )
            pred_scores.append(selected_cls_scores)
            pred_labels.append(selected_cls_scores.new_ones(len(selected)).long() * c)
            pred_boxes.append(selected_cls_bbox)

        pred_scores = torch.cat(pred_scores, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        pred_boxes = torch.cat(pred_boxes, dim=0)

        return pred_scores, pred_labels, pred_boxes

    def _single_class_nms_for_frame(self, cls_score, bbox_pred):
        """
        DO NMS for single class in one frame
        Args:
            cls_score: k. topk scores for one class in one frame
            bbox_pred: k, 7

        Returns:

        """
        # 1. topk scores as k proposals
        # max_scores, _ = cls_score.max(dim=0)
        topk_scores, topk_inds = cls_score.topk(min(self.nms_cfg.NMS_PRE_MAXSIZE, cls_score.size(0)))
        topk_bbox = bbox_pred[topk_inds]
        # 2. get rid of proposals with low scores
        if self.scores_thr is not None:
            scores_mask = (topk_scores >= self.scores_thr)
            topk_scores = topk_scores[scores_mask]
            topk_bbox = topk_bbox[scores_mask]

        # 2. NMS
        selected_inds = []
        if cls_score.shape[0] > 0:
            if self.nms_cfg.NMS_TYPE == 'nms_gpu':
                nms_fun = iou3d_nms_utils.nms_gpu
            elif self.nms_cfg.NMS_TYPE == 'nms_normal_gpu':
                nms_fun = iou3d_nms_utils.nms_normal_gpu
            else:
                raise ValueError('num type not exist!')
            keep_idx, selected_scores = nms_fun(topk_bbox[:, 0:7],
                                                topk_scores,
                                                self.nms_cfg.NMS_THRESH,
                                                **self.nms_cfg
                                                )
            # selected_inds = indices[keep_idx[:self.nms_cfg.NMS_POST_MAXSIZE]]
        return topk_bbox[keep_idx], topk_scores[keep_idx]
