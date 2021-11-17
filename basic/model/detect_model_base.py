#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 上午11:17
# @Author : PH
# @Version：V 0.1
# @File : detect_model_base.py
# @desc :
import torch
import torch.nn as nn
from ..module import voxelize, feature_extractor, roi_head, \
    dense_head, backbone3d, backbone2d, neck, alternative_module_list


class Detect3DBase(nn.Module):

    def __init__(self, top_cfg):
        super().__init__()
        self.top_cfg = top_cfg
        self.model_cfg = top_cfg.MODEL
        self.module_names_ordered = [name.lower() for name in list(self.model_cfg.keys())[1:]]
        self.class_names = top_cfg.DATA_INFO.class_names
        self.num_class = len(self.class_names)
        self.model_info_dict = {
            'module_list': [],
            'training'   : self.training
        }
        data_info = top_cfg.DATA_INFO
        self.model_info_dict.update(data_info)  # add data infos
        self.alternative_module = alternative_module_list

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build_model(self):
        for module_name in self.module_names_ordered:
            # if module_name in self.alternative_module:
            module = getattr(self, f"build_{module_name}")()
            self.add_module(module_name, module)
        return self.model_info_dict['module_list']

    def forward(self, **kwargs):
        raise NotImplementedError

    #########################################################
    def build_voxelize_layer(self):
        if self.model_cfg.get('VOXELIZE_LAYER', None) is None:
            return None
        voxelize_module = voxelize.__all__[self.model_cfg.VOXELIZE_LAYER.NAME](
                model_info_dict=self.model_info_dict,
                **self.model_cfg.VOXELIZE_LAYER
        )
        self.model_info_dict['module_list'].append(voxelize_module)
        return voxelize_module

    def build_feature_extractor(self):
        if self.model_cfg.get('FEATURE_EXTRACTOR', None) is None:
            return None
        feature_extractor_module = feature_extractor.__all__[self.model_cfg.FEATURE_EXTRACTOR.NAME](
                model_info_dict=self.model_info_dict,
                **self.model_cfg.FEATURE_EXTRACTOR
        )
        self.model_info_dict['module_list'].append(feature_extractor_module)
        return feature_extractor_module

    def build_backbone3d(self):
        if self.model_cfg.get('BACKBONE3D', None) is None:
            return None

        backbone3d_module = backbone3d.__all__[self.model_cfg.BACKBONE3D.NAME](
                model_info_dict=self.model_info_dict,
                **self.model_cfg.BACKBONE3D,
        )
        self.model_info_dict['module_list'].append(backbone3d_module)
        return backbone3d_module

    def build_neck(self):
        if self.model_cfg.get('NECK', None) is None:
            return None
        neck_module = neck.__all__[self.model_cfg.NECK.NAME](
                model_info_dict=self.model_info_dict,
                **self.model_cfg.NECK,
        )
        self.model_info_dict['module_list'].append(neck_module)
        return neck_module

    def build_backbone2d(self):
        if self.model_cfg.get('BACKBONE2D', None) is None:
            return None

        backbone2d_module = backbone2d.__all__[self.model_cfg.BACKBONE2D.NAME](
                model_info_dict=self.model_info_dict,
                **self.model_cfg.BACKBONE2D,
        )
        self.model_info_dict['module_list'].append(backbone2d_module)
        return backbone2d_module

    def build_dense_head(self):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None
        dense_head_module = dense_head.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_info_dict=self.model_info_dict,
                top_cfg=self.top_cfg
        )
        self.model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module

    def build_roi_head(self):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None
        point_head_module = roi_head.__all__[self.model_cfg.ROI_HEAD.NAME](
                module_cfg=self.model_cfg.ROI_HEAD,
                model_info_dict=self.model_info_dict,
                # input_channels=model_info_dict['num_point_features'],
                # backbone_channels=model_info_dict['backbone_channels'],
                # point_cloud_range=model_info_dict['point_cloud_range'],
                # voxel_size=model_info_dict['voxel_size'],
                # num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        self.model_info_dict['module_list'].append(point_head_module)
        return point_head_module

    #########################################################
    @staticmethod
    def post_processing(batch_dict, gts, gt_labels, post_cfg):
        """
        """
        from basic.utils.nms_utils import NMS3D
        from basic.metric.average_precision import recall_and_precision, voc_ap, mean_ap

        B = gts.size(0)
        nms_cfg = post_cfg.get('NMS_CONFIG', None)
        eval_cfg = post_cfg.get('EVAL_CONFIG', None)
        frame_ids = batch_dict['frame_id']
        proposal_dict = batch_dict['proposal_dict']

        bboxes = proposal_dict['proposals']
        scores = proposal_dict['proposal_scores']
        final_dict_list = []
        # 1. DO NMS for raw proposals
        if nms_cfg is not None:
            nms = NMS3D(**nms_cfg, confidence_thresh=post_cfg.CONFIDENCE_THRESH)
            nms_result = nms.multi_classes_nms_for_batch(bbox_scores=scores, bbox_preds=bboxes)
            for b in range(B):
                frame_id = frame_ids[b]
                frame_dict = nms_result[b]
                frame_dict.update({'frame_id': frame_id})

                # 2. eval after nms
                eval_dict = {}
                if eval_cfg is not None:
                    if frame_dict['bboxes'].shape[0] > 0:
                        bbox = frame_dict['bboxes']
                        frame_gts = gts[b, :]
                        frame_labels = gt_labels[b, :]
                        real_gts = frame_gts[frame_labels > 0]
                        recall, prec = recall_and_precision(pred_bboxes=bbox, gts=real_gts,
                                                            iou_thresh=eval_cfg.IOU_THRESH
                                                            )
                        # if 'Recall' in eval_cfg.METRIC_NAMES:
                        #     eval_dict.update({'Recall': recall})
                        # if 'Precision' in eval_cfg.METRIC_NAMES:
                        #     eval_dict.update({'Precision': prec})
                        if 'AP' in eval_cfg.METRIC_NAMES:
                            ap = voc_ap(recall, prec)
                            eval_dict.update({'AP': ap})
                        if 'MAP' in eval_cfg.METRIC_NAMES:
                            m_ap = mean_ap(pred_bboxes=bbox, gts=real_gts, iou_threshes=eval_cfg.IOU_THRESHES)
                            eval_dict.update({'MAP': m_ap})
                    else:
                        eval_dict.update({'MAP': 0})  # default key metric is MAP
                        eval_dict.update({'AP': 0})
                frame_dict.update({'eval_dict': eval_dict})
                final_dict_list.append(frame_dict)
        return final_dict_list
