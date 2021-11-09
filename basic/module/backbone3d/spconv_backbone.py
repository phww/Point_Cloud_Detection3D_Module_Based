#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/24 下午4:54
# @Author : PH
# @Version：V 0.1
# @File : spconv_backbone.py
# @desc :
from functools import partial
from .backbone3d_base import VoxelBackBone3D
import spconv
import torch.nn as nn


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None
                   ):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key
                                   )
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(),
    )

    return m


def base_block(in_channel, encoder_channel, encoder_padding, norm_fn, idx):
    # 一层稀疏卷积+两层流形卷积
    blocks = nn.ModuleList()
    blocks.append(post_act_block(in_channel, encoder_channel[0], 3, norm_fn=norm_fn, stride=2,
                                padding=encoder_padding[0], indice_key=f'spconv{idx}', conv_type='spconv'
                                )
                 )
    blocks.append(
            post_act_block(encoder_channel[0], encoder_channel[1], 3, norm_fn=norm_fn, padding=encoder_padding[1],
                           indice_key=f'subm{idx}'
                           )
    )
    blocks.append(
            post_act_block(encoder_channel[1], encoder_channel[2], 3, norm_fn=norm_fn, padding=encoder_padding[2],
                           indice_key=f'subm{idx}'
                           )
    )
    return blocks


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
                inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):

    def __init__(self, model_info_dict,
                 in_channels,
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
                 encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
                 last_pad=None,
                 **kwargs,
                 ):
        super(VoxelBackBone8x, self).__init__()
        # BN
        self.model_info_dict = model_info_dict
        self.input_channels = in_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = model_info_dict['grid_size'][::-1] + [1, 0, 0]  # 41, 1600, 1408 in kitti
        self.conv_input = spconv.SparseSequential(
                spconv.SubMConv3d(self.input_channels, base_channels, 3,
                                  padding=1, bias=False, indice_key='subm1'
                                  ),
                norm_fn(base_channels),
                nn.ReLU(),
        )
        # conv_layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(post_act_block(base_channels, encoder_channels[0][0], 3, norm_fn=norm_fn,
                                               padding=encoder_paddings[0][0], indice_key='subm1'
                                               )
                                )
        # conv2 ~ 4
        in_c = encoder_channels[0][0]
        idx = 2
        for out_channels, padding in zip(encoder_channels[1:], encoder_paddings[1:]):
            self.conv_layers.extend(base_block(in_c, out_channels, padding, norm_fn=norm_fn, idx=idx))
            in_c = out_channels[-1]
            idx += 1
        self.last_pad = 0 if last_pad is None else last_pad
        self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(in_c, output_channels, (3, 1, 1), stride=(2, 1, 1),
                                    padding=self.last_pad,
                                    bias=False, indice_key='spconv_down2'
                                    ),
                norm_fn(output_channels),
                nn.ReLU(),
        )

    def forward(self, data_dict):
        """
        Args:
            voxel_feat:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = data_dict['voxel_features'], data_dict['voxel_coords']
        batch_size = data_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
        )
        # 41, 1600, 1408
        x = self.conv_input(input_sp_tensor)
        # 21, 800, 704 -> 11, 400, 352 -> 5, 200, 176
        for conv in self.conv_layers:
            x = conv(x)

        # for detection head
        # 5, 200, 176 -> 2, 200, 176
        out = self.conv_out(x)

        data_dict['sp_feat3d'] = out

        return data_dict


class VoxelResBackBone8x(VoxelBackBone3D):

    def __init__(self, module_cfg, model_info_dict, **kwargs):
        super(VoxelResBackBone8x, self).__init__(module_cfg, model_info_dict)
        self.module_cfg = module_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = self.grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(
                spconv.SubMConv3d(self.input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(16),
                nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
                SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
                SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
                SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
                SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
                SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
                SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                      conv_type='spconv'
                      ),
                SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
                SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.module_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'
                                    ),
                norm_fn(128),
                nn.ReLU(),
        )
        # some module infos
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        self.down_sample_rate = module_cfg.DOWN_SAMPLE_RATE

    @property
    def output_feature_dims(self):
        return self.num_point_features

    @property
    def output_feature_size(self):
        feature_map_size = self.grid_size // self.down_sample_rate
        feature_map_size[-1] = 2
        return feature_map_size

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor'       : out,
            'encoded_spconv_tensor_stride': 8
        }
        )
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        }
        )

        return batch_dict
