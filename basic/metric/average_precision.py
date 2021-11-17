#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/11 下午2:38
# @Author : PH
# @Version：V 0.1
# @File : average_precision.py
# @desc :
import numpy
import numpy as np
import tables
import torch

from basic.ops.pc_3rd_ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


def recall_and_precision(pred_bboxes, gts, iou_thresh):
    """

    Args:
        pred_bboxes: N, 7
        gts: M, 7
        iou_thresh: list of iou threshold

    Returns:

    """
    num_gt = gts.shape[0]
    num_bbox = pred_bboxes.shape[0]
    if num_gt == 0 or num_bbox == 0:
        recall = 0.
        precision = 0.
        return [recall], [precision]
    ious = boxes_iou3d_gpu(pred_bboxes, gts)  # num_bbox, nmu_gt
    ious = ious.detach().cpu().numpy()
    TP = np.zeros(num_bbox)
    FP = np.zeros(num_bbox)
    used_gt_ind = np.zeros(num_gt)
    for i in range(num_bbox):
        iou = ious[i]
        max_score = iou.max()
        gt_ind = iou.argmax()
        if max_score > iou_thresh and used_gt_ind[gt_ind] == 0:
            TP[i] = 1
            used_gt_ind[gt_ind] = 1
        else:
            FP[i] = 1
    acc_TP = TP.cumsum()
    acc_FP = FP.cumsum()
    recall = acc_TP / num_bbox
    precision = acc_TP / (acc_FP + acc_TP + 1e-8)
    return recall, precision


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 计算面积
    return ap


def mean_ap(pred_bboxes, gts, iou_threshes):
    aps = []
    for iou_thresh in iou_threshes:
        recall, prec = recall_and_precision(pred_bboxes, gts, iou_thresh)
        ap = voc_ap(recall, prec)
        aps.append(ap)
    return np.array(aps).mean()


def multi_classes_mean_ap(pred_bboxes, gts, iou_threshes):
    pass


def print_recall_and_precision(pred_bboxes, gts, iou_threshes):
    r, p = recall_and_precision(pred_bboxes, gts, iou_threshes)
    pass
