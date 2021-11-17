#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/4 下午2:11
# @Author : PH
# @Version：V 0.1
# @File : nms_utils.py
# @desc :
import torch
from ..ops.pc_3rd_ops.iou3d_nms import iou3d_nms_utils


class NMS3D:

    def __init__(self, nms_type, nms_thresh, nms_pre_maxsize=None,
                 nms_post_maxsize=None, confidence_thresh=None, **kwargs
                 ):
        self.nms_type = nms_type
        self.nms_thresh = nms_thresh
        self.k = nms_pre_maxsize
        self.nms_post_maxsize = nms_post_maxsize
        self.scores_thr = confidence_thresh

    def single_class_nms_for_batch(self, cls_scores, bbox_preds):
        """

        Args:
            cls_scores: B, N
            bbox_preds: B, N, 7 + C

        Returns:
            nms_result: M*(frame_ind + bbox(7 + C) + score).
        """
        nms_result = []
        for i, (bbox_pred, cls_score) in enumerate(zip(bbox_preds, cls_scores)):
            frame_dict = {}
            frame_bbox_pred, frame_cls_score = self._single_class_nms_for_frame(cls_score, bbox_pred)
            frame_dict['bboxes'] = frame_bbox_pred
            frame_dict['scores'] = frame_cls_score
            nms_result.append(frame_dict)
        return nms_result

    def multi_classes_nms_for_batch(self, bbox_scores, bbox_preds):
        """
        Args:
            bbox_scores: (B, N, num_class)
            box_preds: (B, N, 7 + C)

        Returns:
            nms_result: M*(frame_ind, bbox(7 + C), score, label)
        """
        B, N, C = bbox_scores.shape[:]
        nms_result = []
        for c in range(1, C):
            cls_scores = bbox_scores[..., c]  # B, N
            nms_result_cls = self.single_class_nms_for_batch(cls_scores=cls_scores,
                                                             bbox_preds=bbox_preds
                                                             )

            for b in range(B):
                frame_dict = {}
                nms_result_frame = nms_result_cls[b]
                cls_labels = torch.ones((nms_result_frame['bboxes'].size(0), 1),
                                        dtype=torch.float, device=bbox_scores.device
                                        ) * c
                if frame_dict.get('bboxes', None) is not None:
                    frame_dict['bboxes'] = torch.cat([frame_dict['bboxes'], nms_result_frame['bboxes']])
                else:
                    frame_dict.update({'bboxes': nms_result_cls[b]['bboxes']})

                if frame_dict.get('scores', None) is not None:
                    frame_dict['scores'] = torch.cat([frame_dict['scores'], nms_result_frame['scores']])
                else:
                    frame_dict.update({'scores': nms_result_frame['scores']})

                if frame_dict.get('labels', None) is not None:
                    frame_dict['labels'] = torch.cat([frame_dict['labels'], cls_labels])
                else:
                    frame_dict.update({'labels': cls_labels})
                nms_result.append(frame_dict)
                # nms_result.append(frame_dict)
            # nms_result_cls = torch.cat((nms_result_cls, cls_labels), dim=-1)  # M*(frame_ind, bbox(7), score, label)
            # nms_result_dict.update(nms_result_cls)
            # nms_result_dict.update(cls_labels)
            # nms_result.append(nms_result_cls)
        # nms_result = torch.cat(nms_result, dim=0)

        return nms_result

    def multi_classes_nms_for_frame(self, bbox_scores, bbox_pred):
        """

        Args:
            bbox_scores: N, num_class
            bbox_pred: N, 7 + C

        Returns:

        """
        N, C = bbox_scores.shape[:]
        final_bboxes, final_scores, final_labels = [], [], []
        for c in range(1, C):
            cls_scores = bbox_scores[:, c]  # N,
            bboxes, scores = self._single_class_nms_for_frame(cls_scores, bbox_pred)
            labels = torch.ones((bboxes.size(0), 1), dtype=torch.float, device=bbox_scores.device) * c
            final_bboxes.append(bboxes)
            final_scores.append(scores)
            final_labels.append(labels)
        final_bboxes = torch.cat(final_bboxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        nms_result = {"bboxes": final_bboxes,
                      "scores": final_scores,
                      "labels": final_labels}
        return nms_result

    def _single_class_nms_for_frame(self, cls_score, bbox_pred):
        """
        DO NMS for single class in one frame
        Args:
            cls_score: N. topk scores for one class in one frame
            bbox_pred: N, 7

        Returns:

        """
        # 1. topk scores as k proposals
        topk_scores, topk_inds = cls_score.topk(min(self.k, cls_score.size(0)))
        topk_bbox = bbox_pred[topk_inds]

        # 2. get rid of proposals with low scores
        if self.scores_thr is not None:
            scores_mask = (topk_scores >= self.scores_thr)
            topk_scores = topk_scores[scores_mask]
            topk_bbox = topk_bbox[scores_mask]

        # 3. DO NMS to figure out candidate index
        selected_inds = []
        if topk_scores.shape[0] > 0:
            if self.nms_type == 'nms_gpu':
                nms_fun = iou3d_nms_utils.nms_gpu
            elif self.nms_type == 'nms_normal_gpu':
                nms_fun = iou3d_nms_utils.nms_normal_gpu
            else:
                raise ValueError('num type not exist!')
            selected_inds = nms_fun(topk_bbox[:, 0:7],
                                    topk_scores,
                                    self.nms_thresh,
                                    self.nms_post_maxsize
                                    )
            if self.nms_post_maxsize is not None:
                selected_inds = selected_inds[:self.nms_post_maxsize]
        return topk_bbox[selected_inds], topk_scores[selected_inds]
