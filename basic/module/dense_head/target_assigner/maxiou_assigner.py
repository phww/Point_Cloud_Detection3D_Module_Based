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
from basic.datatype.assign_result import AssignResult


class MaxIouTargetAssigner(AssignerBase):

    def __init__(self, assigner_cfg, model_info_dict, **kwargs):
        super(MaxIouTargetAssigner, self).__init__(assigner_cfg, model_info_dict, **kwargs)
        self.cfg = assigner_cfg
        self.cls_list = model_info_dict['class_names']
        self.force_match = self.cfg.FORCE_MATCH

    def assign(self, gts, bboxes, gt_labels, **kwargs):
        pos_tuples = []
        neg_tuples = []
        # batch_size == 1, gts:N,
        if gts.dim() == 2:
            return self.assign_one_frame(gts, bboxes, gt_labels, 0)
        else:
            batch_size = gts.size(0)
            # assign target frame by frame
            for i in range(batch_size):
                frame_pos_tuples, frame_neg_tuples = self.assign_one_frame(gts[i], bboxes, gt_labels[i], i)
                if frame_pos_tuples is not None:
                    pos_tuples.append(frame_pos_tuples)
                if frame_neg_tuples is not None:
                    neg_tuples.append(frame_neg_tuples)

            # warning!! maybe a frame have no target.for example,we want to detect car
            # but a frame have no car.Then the pos tuples is empty.

            # num_pos * (batch_id, gt_id + 1, bbox_id) or None
            pos_tuples = torch.cat(pos_tuples).to(self.device) if len(pos_tuples) > 0 else None
            # num_neg * (batch_id, 0, bbox_id)
            neg_tuples = torch.cat(neg_tuples).to(self.device) if len(neg_tuples) > 0 else None

        # 4. encode positive bbox if needed
        pos_bbox_targets = None
        # _, unique_mask = np.unique(pos_tuples[:, (0, 2)].detach().cpu().numpy(), return_index=True, axis=0)
        # pos_tuples = pos_tuples[unique_mask]
        if pos_tuples is not None:
            pos_bboxes = bboxes[pos_tuples[:, 2]]
            pos_bbox_targets = pos_bboxes
            if self.bbox_encoder is not None and pos_bboxes is not None:
                pos_gts = gts[pos_tuples[:, 0], pos_tuples[:, 1]]
                pos_bbox_targets = self.bbox_encoder.encode(boxes=pos_gts, anchors=pos_bboxes)  # num_pos, 7
        return AssignResult(bboxes, gts, gt_labels, pos_bbox_targets, pos_tuples, neg_tuples)

    def assign_one_frame(self, gt, bboxes, gt_labels, batch_id, **kwargs):
        """
        Use one frame ground truth bbox，assign target for all bboxes
        Args:
            real_gt:
            bboxes:
            real_gt_labels:
            **kwargs:

        Returns:

        """
        # 1. calc iou
        ious = self.iou_calculator(gt, bboxes)  # num_gt, num_bbox
        frame_pos_tuples = []
        frame_neg_tuples = []
        # 2.make (batch gt bbox) id tuples according to ious and threshold
        for cls_thr in self.cfg.CLASS_THRESHOLD:  # for every class the threshold may be different
            cls = cls_thr.class_name
            if cls in self.cls_list:
                cls_idx = self.cls_list.index(cls) + 1
                cls_mask = gt_labels == cls_idx
                cls_ious = ious[cls_mask, :]
                pos_thr = cls_thr.get('pos_threshold', 0.7)
                neg_thr = cls_thr.get('neg_threshold', 0.3)
                if cls_ious.size(0) > 0:
                    cls_pos_tuples, cls_neg_tuples = self.make_batch_gt_bbox_tuples(
                            cls_ious, pos_thr, neg_thr, batch_id
                    )
                    if cls_pos_tuples is not None:
                        frame_pos_tuples.append(cls_pos_tuples)
                    if cls_neg_tuples.size(0) > 0:
                        frame_neg_tuples.append(cls_neg_tuples)

        frame_pos_tuples = torch.cat(frame_pos_tuples, dim=0).to(self.device) if len(frame_pos_tuples) > 0 else None
        frame_neg_tuples = torch.cat(frame_neg_tuples, dim=0).to(self.device) if len(frame_neg_tuples) > 0 else None
        # 3. sampling when training
        if self.sampler is not None:
            frame_pos_tuples, frame_neg_tuples = self.sampler(frame_pos_tuples, frame_neg_tuples)

        return frame_pos_tuples, frame_neg_tuples

    def make_batch_gt_bbox_tuples(self, ious, pos_thr, neg_thr, batch_id):
        """
        for every bbox,find the GT with maximum iou.and the (GT index +1 ）with bbox IOU maximum
        """
        # for each gt force to find the bbox with maximum iou. Make sure each gt have a corresponding bbox
        # pos_tuples_force = None
        # if self.force_match:
        #     max_gt, bbox_inds_for_gts = ious.max(dim=1)
        #     pos_gt_ids = torch.arange(ious.size(0))
        #     pos_batch_ids = torch.ones_like(bbox_inds_for_gts) * batch_id
        #     pos_tuples_force = torch.stack([pos_batch_ids.cpu(), pos_gt_ids.cpu(), bbox_inds_for_gts.cpu()],
        #     dim=1).to(
        #             self.device
        #     )
        pos_tuples = None
        neg_tuples = None
        # if i'th bbox maximum iou < neg_threshold,bbox_gt_pairs = (batch_id, -1, i)
        max_bbox, argmax_bbox = ious.max(dim=0)
        neg_bbox_ids = torch.where(max_bbox < neg_thr)[0]
        neg_gt_ids = torch.ones_like(neg_bbox_ids) * -1
        neg_batch_ids = torch.ones_like(neg_bbox_ids) * batch_id
        # num_neg*(batch_id, 0, bbox_id)
        neg_tuples = torch.stack([neg_batch_ids, neg_gt_ids, neg_bbox_ids], dim=1).to(self.device)

        # if i'th bbox maximum iou > pos_threshold,bbox_gt_pairs = (batch_id, Gt index, i)
        pos_bbox_ids = torch.where(max_bbox > pos_thr)[0]
        if pos_bbox_ids.size(0) > 0:
            pos_gt_ids = argmax_bbox[pos_bbox_ids]
            pos_batch_ids = torch.ones_like(pos_bbox_ids) * batch_id
            # num_pos*(batch_id, gt_id, bbox_id)
            pos_tuples = torch.stack([pos_batch_ids, pos_gt_ids, pos_bbox_ids], dim=1).to(self.device)
        # if pos_tuples_force is not None:
        #     # if force match.One bbox in one frame may assigned to different gts.
        #     if pos_tuples is None:
        #         pos_tuples = pos_tuples_force
        #     else:
        #         repeat_assign_mask = []
        #         pos_bbox_id_force = pos_tuples_force[:, 2]
        #         for i, bbox_id in enumerate(pos_bbox_id_force):
        #             if bbox_id not in pos_bbox_ids:
        #                 repeat_assign_mask.append(i)
        #         pos_tuples = torch.cat((pos_tuples,
        #                                 pos_tuples_force[repeat_assign_mask]),
        #                                dim=0
        #                                )
        return pos_tuples, neg_tuples
