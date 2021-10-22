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
    def post_process(self):
        pass
