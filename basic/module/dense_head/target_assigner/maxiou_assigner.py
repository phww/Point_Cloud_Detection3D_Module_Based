#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/8 上午10:29
# @Author : PH
# @Version：V 0.1
# @File : maxiou_assigner.py
# @desc :
import numpy as np
import torch

from .assigner_base import AssignerBase


class MaxIouTargetAssigner(AssignerBase):

    def __init__(self, assigner_cfg, model_info_dict, **kwargs):
        super(MaxIouTargetAssigner, self).__init__(assigner_cfg, model_info_dict, **kwargs)
        self.cfg = assigner_cfg
        self.cls_list = model_info_dict['class_names']

    def assign(self, gts, bboxes, gt_labels, **kwargs):

        pos_tuples = []
        neg_tuples = []
        if gts.dim() == 2:
            return self.assign_one_batch(gts, bboxes, gt_labels, 0)
        else:
            batch_size = gts.size(0)

            for i in range(batch_size):
                batch_pos_tuples, batch_neg_tuples = self.assign_one_batch(gts[i], bboxes, gt_labels[i], i)
                if batch_pos_tuples is not None:
                    pos_tuples.append(batch_pos_tuples)
                if batch_neg_tuples is not None:
                    neg_tuples.append(batch_neg_tuples)
            pos_tuples = torch.cat(pos_tuples).to(self.device) if len(pos_tuples) > 0 else None
            neg_tuples = torch.cat(neg_tuples).to(self.device) if len(neg_tuples) > 0 else None
            # if ret['cls_labels'].size(0) > 0:
            #     cls_labels.append(ret['cls_labels'])
            # if ret['reg_labels'].size(0) > 0:
            #     reg_labels.append(ret['reg_labels'])
            # cls_labels = torch.stack(cls_labels, dim=0).to(self.device)
            # reg_labels = torch.stack(reg_labels, dim=0).to(self.device)

        neg_batch_bbox_id, neg_cls_labels = self.decode_neg_tuples(neg_tuples)
        pos_batch_bbox_id, pos_cls_labels, reg_labels = self.decode_pos_tuples(pos_tuples, gts, gt_labels)
        cls_labels = torch.cat([neg_cls_labels, pos_cls_labels])

        if self.bbox_encoder is not None:
            pos_gts = gts[pos_tuples[:, 0], pos_tuples[:, 1]]
            reg_labels = self.bbox_encoder.encode(reg_labels, pos_gts)

        target_dict = {'cls_labels': cls_labels,
                       'reg_labels': reg_labels
                       }
        batch_bbox_id_dict = {
            'pos': pos_batch_bbox_id,
            'neg': neg_batch_bbox_id
        }
        return target_dict, batch_bbox_id_dict

    def assign_one_batch(self, gts, bboxes, gt_labels, batch_id, **kwargs):
        """
        Use one batch(frame) ground truth bbox，assign target for all bboxes
        Args:
            gts:
            bboxes:
            gt_labels:
            **kwargs:

        Returns:

        """
        # 1. calc iou
        ious = self.iou_calculator(bboxes, gts).T  # num_gt, num_bbox
        batch_pos_tuples = []
        batch_neg_tuples = []
        # 2.make batch gt bbox tuples according to ious and threshold
        for cls_thr in self.cfg.CLASS_THRESHOLD:  # for every class the threshold maybe different
            cls = cls_thr.class_name
            if cls in self.cls_list:
                index = self.cls_list.index(cls) + 1
                gt_mask = gt_labels == index
                cls_ious = ious[gt_mask, :]
                pos_thr = cls_thr.get('pos_threshold', 0.7)
                neg_thr = cls_thr.get('neg_threshold', 0.3)
                cls_pos_tuples, cls_neg_tuples = self.make_batch_gt_bbox_tuples(cls_ious, pos_thr, neg_thr, batch_id)
                if cls_pos_tuples.size(0) > 0:
                    batch_pos_tuples.append(cls_pos_tuples)
                if cls_neg_tuples.size(0) > 0:
                    batch_neg_tuples.append(cls_neg_tuples)

        batch_pos_tuples = torch.cat(batch_pos_tuples, dim=0).to(self.device) if len(batch_pos_tuples) > 0 else None
        batch_neg_tuples = torch.cat(batch_neg_tuples, dim=0).to(self.device) if len(batch_neg_tuples) > 0 else None
        if self.sampler is not None:
            batch_pos_tuples, batch_neg_tuples = self.sampler(batch_pos_tuples, batch_neg_tuples)
        # # 3.assign class labels for every bbox.
        # cls_labels = bbox_gt_pairs
        # pos_mask = cls_labels > 1
        # # convert pos bbox from gt index + 1 to gt labels
        # cls_labels[pos_mask] = gt_labels[(cls_labels[pos_mask] - 1)].long()
        #
        # # 4.assign regression labels for every positive bbox.
        # reg_labels = bboxes[pos_mask]
        #
        # targets_dict = {
        #     "cls_labels": cls_labels,
        #     "reg_labels": reg_labels,
        # }
        return batch_pos_tuples, batch_neg_tuples

    def make_batch_gt_bbox_tuples(self, ious, pos_thr, neg_thr, batch_id, ):
        # for every bbox,find the GT with maximum iou.and the (GT index +1 ）with bbox IOU maximum
        # if i'th bbox maximum iou < neg_threshold,bbox_gt_pairs = (batch_id, i, 0)
        # if i'th bbox maximum iou > pos_threshold,bbox_gt_pairs = (batch_id, i, Gt index + 1)
        max_bbox, argmax_bbox = ious.max(dim=0)
        neg_bbox_ids = torch.where(max_bbox < neg_thr)[0]
        neg_gt_ids = torch.zeros_like(neg_bbox_ids)
        neg_batch_ids = torch.ones_like(neg_bbox_ids) * batch_id
        neg_tuples = torch.stack([neg_batch_ids, neg_gt_ids, neg_bbox_ids], dim=1).to(self.device
                                                                                      )  # (batch_id, 0, bbox_id)

        pos_bbox_ids = torch.where(max_bbox > pos_thr)[0]
        pos_gt_ids = argmax_bbox[pos_bbox_ids] + 1
        pos_batch_ids = torch.ones_like(pos_bbox_ids) * batch_id
        pos_tuples = torch.stack([pos_batch_ids, pos_gt_ids, pos_bbox_ids], dim=1).to(self.device
                                                                                      )  # (batch_id, gt_id +1, bbox_id)

        return pos_tuples, neg_tuples

    def decode_neg_tuples(self, tuples):
        neg_batch_bbox_id = torch.unique(tuples[:, [0, 2]], dim=0).to(self.device)  # N*(batch_id, bbox_id)
        neg_cls_labels = torch.zeros(neg_batch_bbox_id.shape[0], device=self.device)
        return neg_batch_bbox_id, neg_cls_labels

    def decode_pos_tuples(self, tuples, gts, gt_labels):
        pos_batch_bbox_id = torch.unique(tuples[:, [0, 2]], dim=0).to(self.device)
        pos_gt_bbox_id = tuples[:, [1, 2]]
        pos_cls_labels = gt_labels[tuples[:, 0], tuples[:, 1] - 1]
        pos_reg_labels = gts[tuples[:, 0], tuples[:, 1] - 1]
        return pos_batch_bbox_id, pos_cls_labels, pos_reg_labels
