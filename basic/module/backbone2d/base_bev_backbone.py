import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, module_cfg, model_info_dict):
        super().__init__()
        self.module_cfg = module_cfg
        input_channels = model_info_dict['cur_point_feature_dims']
        # down sample config
        if self.module_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.module_cfg.LAYER_NUMS) == len(self.module_cfg.LAYER_STRIDES)\
                   == len(self.module_cfg.LAYER_OUT_CHANNELS)
            layer_nums = self.module_cfg.LAYER_NUMS
            layer_strides = self.module_cfg.LAYER_STRIDES
            layer_out_channels = self.module_cfg.LAYER_OUT_CHANNELS
        else:
            layer_nums = layer_strides = layer_out_channels = []

        # up sample config
        if self.module_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.module_cfg.UPSAMPLE_STRIDES) == len(self.module_cfg.UPSAMPLE_CHANNELS)
            unpsample_channels = self.module_cfg.UPSAMPLE_CHANNELS
            upsample_strides = self.module_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = unpsample_channels = []

        num_levels = len(layer_nums)
        layer_in_channels = [input_channels, *layer_out_channels[:-1]]

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    layer_in_channels[idx], layer_out_channels[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(layer_out_channels[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            # same conv
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(layer_out_channels[idx], layer_out_channels[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(layer_out_channels[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.down_blocks.append(nn.Sequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.up_blocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            layer_out_channels[idx], unpsample_channels[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(unpsample_channels[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.up_blocks.append(nn.Sequential(
                        nn.Conv2d(
                            layer_out_channels[idx], unpsample_channels[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(unpsample_channels[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(unpsample_channels)
        if len(upsample_strides) > num_levels:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        # 256, 200 , 176 -> 128, 200, 176 -> 256, 100, 88
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.up_blocks) > 0:
                ups.append(self.up_blocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.up_blocks) > len(self.down_blocks):
            x = self.up_blocks[-1](x)

        data_dict['dense_feat_2d'] = x

        return data_dict

    @property
    def output_feature_dims(self):
        return self.num_bev_features
