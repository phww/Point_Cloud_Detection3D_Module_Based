#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/23 上午11:17
# @Author : PH
# @Version：V 0.1
# @File : detect_model_base.py
# @desc :
import torch
import torch.nn as nn
from ..module import feature_extractor, roi_head, dense_head, backbone3d, backbone2d, neck, alternative_module_list
from basic.utils.nms_utils import class_agnostic_nms, multi_classes_nms

class Detect3DBase(nn.Module):

    def __init__(self, model_cfg, data_infos):
        super().__init__()
        self.model_cfg = model_cfg.MODEL
        self.module_names_ordered = [name.lower() for name in list(self.model_cfg.keys())[1:]]
        self.num_class = len(data_infos['class_names'])
        self.class_names = data_infos['class_names']
        self.model_info_dict = {
            'module_list': [],
            'training': self.training
        }
        self.model_info_dict.update(data_infos)  # add data infos
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        # 'map_to_bev_module', 'pfe'
        self.alternative_module = alternative_module_list

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build_model(self):
        for module_name in self.module_names_ordered:
            if module_name in self.alternative_module:
                module = getattr(self, f"build_{module_name}")()
                self.add_module(module_name, module)
        return self.model_info_dict['module_list']

    def forward(self, **kwargs):
        raise NotImplementedError

    #########################################################
    def build_feature_extractor(self):
        if self.model_cfg.get('FEATURE_EXTRACTOR', None) is None:
            return None
        feature_extractor_module = feature_extractor.__all__[self.model_cfg.FEATURE_EXTRACTOR.NAME](
                module_cfg=self.model_cfg.FEATURE_EXTRACTOR,
                model_info_dict=self.model_info_dict,
        )
        self.model_info_dict['module_list'].append(feature_extractor_module)
        self.model_info_dict['cur_point_feature_dims'] = feature_extractor_module.output_feature_dims
        return feature_extractor_module

    def build_backbone3d(self):
        if self.model_cfg.get('BACKBONE3D', None) is None:
            return None

        backbone3d_module = backbone3d.__all__[self.model_cfg.BACKBONE3D.NAME](
                module_cfg=self.model_cfg.BACKBONE3D,
                model_info_dict=self.model_info_dict,
        )
        self.model_info_dict['module_list'].append(backbone3d_module)
        self.model_info_dict['cur_point_feature_dims'] = backbone3d_module.output_feature_dims
        # self.model_info_dict['backbone_channels'] = backbone3d_module.backbone_channels \
        #     if hasattr(backbone3d_module, 'backbone_channels') else None
        self.model_info_dict['feature_map_size'] = backbone3d_module.output_feature_size
        return backbone3d_module

    def build_neck(self):
        if self.model_cfg.get('NECK', None) is None:
            return None
        neck_module = neck.__all__[self.model_cfg.NECK.NAME](
                module_cfg=self.model_cfg.NECK,
                model_info_dict=self.model_info_dict,
        )
        self.model_info_dict['module_list'].append(neck_module)
        self.model_info_dict['cur_point_feature_dims'] = neck_module.output_feature_dims
        self.model_info_dict['feature_map_size'] = neck_module.output_feature_size
        return neck_module

    def build_backbone2d(self):
        if self.model_cfg.get('BACKBONE2D', None) is None:
            return None

        backbone2d_module = backbone2d.__all__[self.model_cfg.BACKBONE2D.NAME](
                module_cfg=self.model_cfg.BACKBONE2D,
                model_info_dict=self.model_info_dict,
        )
        self.model_info_dict['module_list'].append(backbone2d_module)
        self.model_info_dict['cur_point_feature_dims'] = backbone2d_module.num_bev_features
        return backbone2d_module

    def build_dense_head(self):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None
        dense_head_module = dense_head.__all__[self.model_cfg.DENSE_HEAD.NAME](
                module_cfg=self.model_cfg.DENSE_HEAD,
                model_info_dict=self.model_info_dict,
                # num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                # class_names=self.class_names,
                # grid_size=model_info_dict['grid_size'],
                # point_cloud_range=model_info_dict['point_cloud_range'],
                # predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
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
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class + 1]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = multi_classes_nms(
                            cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                            nms_config=post_process_cfg.NMS_CONFIG,
                            score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds
                selected, selected_scores = class_agnostic_nms(
                        box_scores=cls_preds, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            # recall_dict = self.generate_recall_record(
            #         box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
            #         recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
            #         thresh_list=post_process_cfg.RECALL_THRESH_LIST
            # )

            record_dict = {
                'pred_boxes' : final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict
