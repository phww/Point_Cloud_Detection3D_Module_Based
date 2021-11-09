#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/27 下午4:31
# @Author : PH
# @Version：V 0.1
# @File : anchor_gen_base.py
# @desc :
import easydict
import numpy as np
import torch
import einops


# todo: base anchor generator 、2d anchor
# help fun
def _add_size_info(anchors_without_size, sizes):
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(anchors_without_size)):
        # torch.repeat == np.tail
        anchors_without_size[i] = anchors_without_size[i].unsqueeze(dim=-2).repeat(tile_shape)
        anchors_without_size[i] = anchors_without_size[i].unsqueeze(dim=-1)  # for concat
    sizes = torch.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(anchors_without_size[0].shape)
    tile_size_shape[3] = 1
    sizes = sizes.repeat(tile_size_shape)
    anchors_without_size.insert(3, sizes)
    # gen anchors
    anchors = torch.cat(anchors_without_size, dim=-1)  # x y z w=dx l=dy h=dz r
    return anchors


class AnchorGenerator:

    def __init__(self, anchor_gen_cfg, model_info_dict, class_type='Car', dtype=torch.float32, **kwargs
                 ):
        """
        Anchor Generator for one class and feature_map_size.
        Args:
            anchor_gen_cfg:(Easydict)
            model_info_dict: (dict) extra information for the entire program
            feature_map_size: (torch shape) or (list of int)
                                D H W for anchor3d H W for anchor2d
            class_type: (str)
                        one class type -> one anchor generator
            dtype: torch dtype

        """
        self.anchor_cfgs = anchor_gen_cfg
        self.anchor_cfg = None
        for cfg in self.anchor_cfgs.CLASS_CONFIG:
            if cfg.class_name == class_type:
                self.anchor_cfg = cfg
                break
        if self.anchor_cfg is None:
            raise ValueError(f"Can Not Find Anchor Config For Class:{class_type}")
        # basic
        self.anchor_dims = self.anchor_cfg.anchor_dims
        self.boxes_size = self.anchor_cfg['boxes_size']
        self.ratios = self.anchor_cfg.get('ratios', [1])
        self.rots = self.anchor_cfg.get('rotations', [0])
        self.center_aligned = self.anchor_cfg.get('center_aligned', False)
        # if only need anchors on road.anchors z = road_plane_height and z_dim=1
        self.road_plane_aligned = self.anchor_cfg.get('road_plane_aligned', False)
        if self.road_plane_aligned:
            # the height is based on pint cloud coordinate.set it in cfg if needed
            self.road_plane_height = self.anchor_cfg.get('road_plane_height', 0)
        #     # for 3d anchor feature_map_size is D H W. for 2d anchor feature_map_size is 1 H W.
        # if len(self.feature_map_size) == 2:
        #     self.feature_map_size.insert(0, 1)
        self.dtype = dtype

        self.mode = self.anchor_cfg['mode']
        # Range Mode needed
        self.extents_range = anchor_gen_cfg.get('range', None)
        if self.extents_range is None:  # if don't set stride in cfg,for 3d anchor the range is point cloud cube range
            if self.anchor_dims == 3:
                self.extents_range = model_info_dict['point_cloud_range']
            # for 2d anchor the range is image size [0, 0, H , W]
            else:
                self.extents_range = model_info_dict['image_size_range']
                self.extents_range.insert(0, 3)
            # X_len, Y_len, Z_len
            self.extents_len = torch.tensor(self.extents_range, dtype=self.dtype)[self.anchor_dims:] - \
                               torch.tensor(self.extents_range, dtype=self.dtype)[:self.anchor_dims]

        # Stride Mode needed
        self.stride = self.anchor_cfg.get('stride', None)

        # grid_X, grid_Y, grid_Z
        self.grid_size = model_info_dict.get('grid_size', None)
        self.voxel_size = model_info_dict.get('voxel_size', None)
        #
        self.device = anchor_gen_cfg.DEVICE
        # self.extra_value_names = self.anchor_cfg.get('extra_value_names', None)
        # self._gen_sizes_with_ratios()

    def gen_anchors(self, feature_map_size, flatten_output=True):
        if self.stride is None:  # if don't set stride in cfg,auto set stride
            # for 3d anchor extents_len is X_len, Y_len, Z_len. for 2d anchor extents_len is X_len, Y_len, 0.
            if self.anchor_dims == 2:
                self.extents_len.insert(0, 2)
            shape = np.array(feature_map_size, np.float32)[[2, 1, 0]]  # W H D
            self.stride = self.extents_len / shape  # x_stride y_stride z_stride/0

        if self.mode == 'Base_Anchor':
            # anchors = self._gen_anchor_from_base()
            anchors = None
        elif self.mode == 'Range':
            anchors = self._gen_anchor_range(feature_map_size)
            if flatten_output:
                anchors = anchors.view(-1, self.ndim)
        elif self.mode == 'Stride':
            anchors = self._gen_anchor_stride(feature_map_size)
            if flatten_output:
                anchors = anchors.view(-1, self.ndim)
        else:
            raise ValueError("Unsupported Mode!")
        return anchors.to(self.device)

    def _gen_sizes_with_ratios(self):
        sizes = np.array(self.boxes_size, dtype=self.dtype)  # N, 3
        ratios = np.array(self.ratios, dtype=self.dtype)  # K,

    def _gen_base_anchors(self):
        pass

    def _gen_anchor_from_base(self):
        pass

    def _gen_anchor_stride(self, feature_map_size):
        assert self.stride is not None, "stride needed"
        x_stride, y_stride, z_stride = self.stride
        x_start, y_start, z_start = self.extents_range[:self.anchor_dims]
        # D, H, W == Z Y X
        z_centers = torch.arange(feature_map_size[0], dtype=self.dtype)  # [0, D]
        y_centers = torch.arange(feature_map_size[1], dtype=self.dtype)  # [0, H]
        x_centers = torch.arange(feature_map_size[2], dtype=self.dtype)  # [0, W]
        z_centers = z_centers * z_stride + z_start
        y_centers = y_centers * y_stride + y_start
        x_centers = x_centers * x_stride + x_start

        if self.center_aligned:
            x_centers += self.stride[0] / 2
            y_centers += self.stride[1] / 2
            z_centers += self.stride[2] / 2
        if self.road_plane_aligned:
            assert feature_map_size[0] == 1, "z_dims should equal to 1"
            z_centers = torch.tensor(self.road_plane_height, dtype=self.dtype)
        # list to torch tensor
        sizes = torch.tensor(self.boxes_size, dtype=self.dtype)
        rotations = torch.tensor(self.rots, dtype=self.dtype)
        anchors_without_size = list(torch.meshgrid(x_centers, y_centers, z_centers, rotations))
        # put size info to anchors_without_size
        anchors = _add_size_info(anchors_without_size, sizes)  # x y z w=dx l=dy h=dz r
        if self.anchor_dims == 2:
            anchors = anchors[:, :, :, 0, :, 0, :].squeeze()  # x y h w r
        return anchors

    def _gen_anchor_range(self, feature_map_size):
        assert self.extents_range is not None, "range needed"
        # feature_map_size D H W -> Z Y X
        extents_range = torch.tensor(self.extents_range, dtype=self.dtype)
        # voxel center alignment
        if self.center_aligned:
            extents_range[:3] = extents_range[:3] + self.stride / 2
            extents_range[3:] = extents_range[3:] - self.stride / 2

        x_centers = torch.linspace(
            extents_range[0], extents_range[3],
            feature_map_size[2], dtype=self.dtype
        )
        y_centers = torch.linspace(
            extents_range[1], extents_range[4],
            feature_map_size[1], dtype=self.dtype
        )
        z_centers = torch.linspace(
            extents_range[2], extents_range[5],
            feature_map_size[0], dtype=self.dtype
        )
        if self.road_plane_aligned:
            assert feature_map_size[0] == 1, "z_dims should equal to 1"
            z_centers = torch.tensor(self.road_plane_height, dtype=self.dtype)
        sizes = torch.tensor(self.boxes_size, dtype=self.dtype)
        rotations = torch.tensor(self.rots, dtype=self.dtype)
        anchors_without_size = list(torch.meshgrid(x_centers, y_centers, z_centers, rotations))
        # put size info to anchors_without_size
        anchors = _add_size_info(anchors_without_size, sizes)  # x y z w=dx l=dy h=dz r
        if self.anchor_dims == 2:
            anchors = anchors[:, :, 0, :, :, 0, :].squeeze()  # x y h w r
        return anchors

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self.rots)
        num_size = len(self.boxes_size)
        num_ratio = len(self.ratios)
        return num_rot * num_size * num_ratio

    @property
    def ndim(self):
        return 7  # + len(self.extra_values)

    # @property
    # def shape(self, DHW=False):
    #     num_rot = len(self.rots)
    #     num_size = len(self.boxes_size)
    #     num_ratio = len(self.ratios)
    #     shape = []
    #     feature_map_size = self.feature_map_size.copy()
    #     # DHW -> WHD
    #     if not DHW:
    #         feature_map_size[0], feature_map_size[2] = feature_map_size[2], feature_map_size[0]
    #     shape.extend(feature_map_size)
    #     shape.append(num_size)
    #     shape.append(num_rot)
    #     return shape

    # for debug
    def set_mode(self, mode_str):
        assert mode_str in ['Range', 'Stride', 'Base_Anchor']
        self.mode = mode_str

    def set_range(self, range):
        assert isinstance(range, list)
        self.extents_range = range

    def set_stride(self, stride):
        assert isinstance(stride, list)
        self.stride = stride

    def set_anchor_dims(self, dims):
        self.anchor_dims = dims


class MultiClsAnchorGenerator:

    def __init__(self, anchor_gen_cfg, model_info_dict, feature_map_size=None, cls_list=None, dtype=torch.float32):
        _feature_map_size = model_info_dict['feature_map_size'] if feature_map_size is None else feature_map_size
        _cls_list = model_info_dict['class_names'] if cls_list is None else cls_list
        self.anchor_generator_list = [AnchorGenerator(anchor_gen_cfg, model_info_dict, cls, _feature_map_size, dtype)
                                      for cls in _cls_list]

    def gen_anchors(self, flatten_output=True):
        all_anchors = []
        for anchor_generator in self.anchor_generator_list:
            anchors = anchor_generator.gen_anchors(flatten_output)
            all_anchors.append(anchors.unsqueeze(dim=0))
        all_anchors = torch.cat(all_anchors, dim=0)
        if flatten_output:
            all_anchors = all_anchors.view(-1, self.ndim)
        return all_anchors

    @property
    def num_anchors_per_localization(self):
        num = 0
        for anchor_generator in self.anchor_generator_list:
            num += anchor_generator.num_anchors_per_localization
        return num

    @property
    def ndim(self):
        return 7
