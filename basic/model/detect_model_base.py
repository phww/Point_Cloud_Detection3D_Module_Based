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


# from basic.utils.nms_utils import class_agnostic_nms, multi_classes_nms


class Detect3DBase(nn.Module):

    def __init__(self, top_cfg, data_infos):
        super().__init__()
        self.top_cfg = top_cfg
        self.model_cfg = top_cfg.MODEL
        self.module_names_ordered = [name.lower() for name in list(self.model_cfg.keys())[1:]]
        self.num_class = len(data_infos['class_names'])
        self.class_names = data_infos['class_names']
        self.model_info_dict = {
            'module_list': [],
            'training'   : self.training
        }
        self.model_info_dict.update(data_infos)  # add data infos
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
    def post_processing(self, proposal_dict, post_cfg=None):
        """
        """
        from basic.utils.nms_utils import multi_classes_nms, single_class_nms

        if post_cfg is None:
            post_process_cfg = self.model_cfg.INFERENCE_CONFIG.POST_PROCESSING
        else:
            post_process_cfg = post_cfg
        nms_cfg = post_process_cfg.get('NMS_CONFIG', None)
        bboxes = proposal_dict['proposals']
        scores = proposal_dict['proposal_scores']

        final_bboxes, final_scores = [], []
        if nms_cfg is not None:
            for bbox, score in zip(bboxes, scores):
                if nms_cfg.MULTI_CLASSES_NMS:
                    selected_inds, selected_scores = multi_classes_nms(cls_scores=score,
                                                                       box_preds=bbox,
                                                                       nms_config=nms_cfg,
                                                                       score_thresh=post_process_cfg.SCORE_THRESH
                                                                       )
                else:
                    selected_inds, selected_scores = single_class_nms(cls_scores=score,
                                                                      bbox_preds=bbox,
                                                                      nms_config=nms_cfg,
                                                                      score_thresh=post_process_cfg.SCORE_THRESH
                                                                      )
                selected_bbox = bbox[selected_inds]
                final_bboxes.append(selected_bbox)
                final_scores.append(selected_scores)
            final_bboxes = torch.cat(final_bboxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
        pred_dict = {
            'pred_bboxes': final_bboxes,
            'pred_scores': final_scores
        }
        return pred_dict
