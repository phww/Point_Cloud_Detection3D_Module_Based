#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/1 下午2:49
# @Author : PH
# @Version：V 0.1
# @File : assign_result.py
# @desc :
import numpy as np
import torch


class AssignResult:

    def __init__(self, bboxes, gts, gt_labels, pos_bbox_targets, pos_tuples, neg_tuples):
        """
        Assign result for all bboxes/anchors.If a bbox is not sampled for calculate loss.
        the targets weights of it is 0, otherwise 1.
        Args:
            bboxes[tensor]: A, 7
                bboxes/anchors produced by anchor generator
            gts[tensor]: B, M
            gt_labels[tensor]: B, M
            pos_bbox_targets[tensor]: num_pos, 7
                positive bbox/anchor's targets for bbox regression
            pos_tuples[tensor]: num_pos, 3
                positive bboxes/anchors tuples. [batch_ind, gt_ind, bbox_ind]
            neg_tuples: num_neg, 3
                negative bboxes/anchors tuples. [batch_ind, -1, bbox_ind]
        """
        # unique_mask = pos_tuples[:, (0, 2)].unique(dim=0, return_index=True)

        self.pos_tuples = pos_tuples
        self.neg_tuples = neg_tuples
        self.device = gts.device
        self.bboxes = bboxes
        self.batch_size = gts.size(0)
        self.num_bbox = bboxes.size(0)

        self.pos_tuples_dense = self.dense_repr(pos_tuples)
        # bbox_targets for background is 0 vector.foreground is pos_bbox_targets
        self.bbox_targets = self._get_bbox_targets(pos_bbox_targets)
        # cls_targets of background is 0. foreground is the corresponding gt's label
        self.cls_targets = self._get_cls_targets(gt_labels)
        # bbox weights of foreground is 1. others is 0
        bbox_weights = torch.zeros_like(self.pos_tuples_dense)
        bbox_weights[self.pos_tuples_dense >= 0] = 1.0
        self.bbox_weights = bbox_weights
        # cls_weights used in cls_loss.means whether a bbox is used in cls_loss
        self.cls_weights = self._get_cls_weights()

    def _get_bbox_targets(self, pos_bbox_targets):
        bbox_targets = torch.zeros((self.batch_size, self.num_bbox, 7), device=self.device, dtype=torch.float)
        if self.pos_tuples is not None:
            ind = torch.where(self.pos_tuples_dense >= 0)
            torch.index_put_(bbox_targets, indices=ind, values=pos_bbox_targets)
        return bbox_targets

    def _get_cls_targets(self, gt_labels):
        bbox_labels = torch.zeros_like(self.pos_tuples_dense, device=self.device, dtype=torch.long)
        if self.pos_tuples is not None:
            ind = torch.where(self.pos_tuples_dense >= 0)
            val = gt_labels[self.pos_tuples[:, 0], self.pos_tuples[:, 1]].long()
            torch.index_put_(bbox_labels, indices=ind, values=val)
        return bbox_labels

    def _get_cls_weights(self):
        if self.pos_tuples is not None:
            tuples = torch.cat([self.pos_tuples, self.neg_tuples], dim=0)
        else:
            tuples = self.neg_tuples
        tuples_dense = self.dense_repr(tuples)
        cls_weights = torch.zeros_like(tuples_dense, device=self.device)
        cls_weights[tuples_dense != -100] = 1.0
        return cls_weights

    def add_gts(self, gts, gt_labels):
        num_gt = gts.size(1)
        self.bbox_targets = torch.cat((self.bbox_targets, gts), dim=1)
        gt_weights = torch.zeros_like(gt_labels, device=self.device, dtype=torch.float)
        gt_weights[gt_labels > 0] = 1.0
        self.bbox_weights = torch.cat((self.bbox_weights, gt_weights), dim=1)
        self.cls_targets = torch.cat((self.cls_targets, gt_labels), dim=1)
        gt_weights = torch.zeros_like(gt_labels, device=self.device, dtype=torch.float)
        gt_weights[gt_labels > 0] = 1.0
        self.cls_weights = torch.cat((self.cls_weights, gt_weights), dim=1)

    def dense_repr(self, tuples, gt_labels=None, defualt_value=-100):
        """
        dense representation or [batch, gt, bbox] tuples.
        Args:
            tuples: [batch, gt, bbox] tuples
            raw_size: total num of bbox/anchors
            batch_size: B

        Returns:
            dense representation: B, num_bbox

        """
        dense_ret = torch.ones(self.batch_size, self.num_bbox, dtype=torch.long, device=self.device) * defualt_value
        if tuples is not None:
            if tuples.dtype != torch.long:
                tuples = tuples.long()
            # default value is -100
            ind = (tuples[:, 0], tuples[:, 2])
            val = tuples[:, 1]
            if gt_labels is not None:
                val = torch.gather(gt_labels, dim=0, index=tuples[:, :1])
            torch.index_put_(dense_ret, indices=ind, values=val)
        return dense_ret

    @property
    def pos_bboxes(self):
        return self.bbox_targets

    @property
    def neg_bboxes(self):
        return self.bboxes

    @property
    def pos_labels(self):
        return self.bboxes

    @property
    def neg_labels(self):
        return self.bboxes
