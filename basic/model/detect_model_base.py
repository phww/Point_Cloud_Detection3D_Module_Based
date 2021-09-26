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

    def __init__(self, model_cfg, num_class, data_infos):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = data_infos['class_names']
        del data_infos['class_names']
        self.data_infos = data_infos
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        # 'map_to_bev_module', 'pfe'
        self.alternative_module = alternative_module_list

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build_model(self):
        model_info_dict = {
            'module_list': [],
        }
        model_info_dict.update(self.data_infos)

        for module_name in self.alternative_module:
            module, model_info_dict = getattr(self, f"build_{module_name}")(
                    model_info_dict
            )
            self.add_module(module_name, module)

    def forward(self, **kwargs):
        raise NotImplementedError

    #########################################################
    def build_feature_extractor(self, model_info_dict):
        if self.model_cfg.get('FEATURE_EXTRACTOR', None) is None:
            return None, model_info_dict
        feature_extractor_module = feature_extractor.__all__[self.model_cfg.VFE.NAME](
                module_cfg=self.model_cfg.FEATURE_EXTRACTOR,
                **model_info_dict,
                # num_point_features=model_info_dict['raw_point_feature_dims'],
                # point_cloud_range=model_info_dict['point_cloud_range'],
                # voxel_size=model_info_dict['voxel_size'],
                # grid_size=model_info_dict['grid_size'],
                # depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['cur_point_feature_dims'] = feature_extractor_module.get_output_feature_dim()
        model_info_dict['module_list'].append(feature_extractor_module)
        return feature_extractor_module, model_info_dict

    def build_backbone3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE3D', None) is None:
            return None, model_info_dict

        backbone3d_module = backbone3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
                model_cfg=self.model_cfg.BACKBONE_3D,
                **model_info_dict,
                # input_channels=model_info_dict['num_point_features'],
                # grid_size=model_info_dict['grid_size'],
                # voxel_size=model_info_dict['voxel_size'],
                # point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone3d_module)
        model_info_dict['cur_point_feature_dims'] = backbone3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone3d_module.backbone_channels \
            if hasattr(backbone3d_module, 'backbone_channels') else None
        return backbone3d_module, model_info_dict

    def build_neck(self, model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = neck.__all__[self.model_cfg.NECK.NAME](
                model_cfg=self.model_cfg.NECK,
                model_info_dict=model_info_dict,
        )

    def build_backbone2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE2D', None) is None:
            return None, model_info_dict

        backbone2d_module = backbone2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
                module_cfg=self.model_cfg.BACKBONE_2D,
                model_info_dict=model_info_dict,
                # input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone2d_module)
        model_info_dict['num_bev_features'] = backbone2d_module.num_bev_features
        return backbone2d_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_head.__all__[self.model_cfg.DENSE_HEAD.NAME](
                module_cfg=self.model_cfg.DENSE_HEAD,
                model_info_dict=model_info_dict,
                # num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                # class_names=self.class_names,
                # grid_size=model_info_dict['grid_size'],
                # point_cloud_range=model_info_dict['point_cloud_range'],
                # predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_head.__all__[self.model_cfg.ROI_HEAD.NAME](
                model_cfg=self.model_cfg.ROI_HEAD,
                input_channels=model_info_dict['num_point_features'],
                backbone_channels=model_info_dict['backbone_channels'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                voxel_size=model_info_dict['voxel_size'],
                num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    #########################################################
    def post_process(self):
        pass
