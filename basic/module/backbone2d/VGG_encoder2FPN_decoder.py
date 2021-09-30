#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/26 下午4:30
# @Author : PH
# @Version：V 0.1
# @File : VGG_encoder2FPN_decoder.py
# @desc :
import torch
import torch.nn as nn


class VggEncoder(nn.Module):

    def __init__(self, in_channel, conv_channels, repeats):
        super(VggEncoder, self).__init__()
        self.conv1 = nn.Sequential()
        for i in range(repeats[0]):
            if i == 0:
                self.conv1.add_module(f"conv1-{i}", nn.Conv2d(in_channel, conv_channels[0], (3, 3), padding=(1, 1)))
                self.conv1.add_module(f"bn1-{i}", nn.BatchNorm2d(conv_channels[0]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())
            else:
                self.conv1.add_module(f"conv1-{i}",
                                      nn.Conv2d(conv_channels[0], conv_channels[0], (3, 3), padding=(1, 1))
                                      )
                self.conv1.add_module(f"bn1-{i}", nn.BatchNorm2d(conv_channels[0]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())

        self.conv2 = nn.Sequential()
        for i in range(repeats[1]):
            if i == 0:
                self.conv2.add_module(f"conv2-{i}",
                                      nn.Conv2d(conv_channels[0], conv_channels[1], (3, 3), padding=(1, 1))
                                      )
                self.conv2.add_module(f"bn2-{i}", nn.BatchNorm2d(conv_channels[1]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())
            else:
                self.conv2.add_module(f"conv2-{i}",
                                      nn.Conv2d(conv_channels[1], conv_channels[1], (3, 3), padding=(1, 1))
                                      )
                self.conv2.add_module(f"bn2-{i}", nn.BatchNorm2d(conv_channels[1]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())

        self.conv3 = nn.Sequential()
        for i in range(repeats[2]):
            if i == 0:
                self.conv3.add_module(f"conv3-{i}",
                                      nn.Conv2d(conv_channels[1], conv_channels[2], (3, 3), padding=(1, 1))
                                      )
                self.conv3.add_module(f"bn3-{i}", nn.BatchNorm2d(conv_channels[2]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())
            else:
                self.conv3.add_module(f"conv3-{i}",
                                      nn.Conv2d(conv_channels[2], conv_channels[2], (3, 3), padding=(1, 1))
                                      )
                self.conv3.add_module(f"bn3-{i}", nn.BatchNorm2d(conv_channels[2]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())

        self.conv4 = nn.Sequential()
        for i in range(repeats[3]):
            if i == 0:
                self.conv4.add_module(f"conv4-{i}",
                                      nn.Conv2d(conv_channels[2], conv_channels[3], (3, 3), padding=(1, 1))
                                      )
                self.conv4.add_module(f"bn4-{i}", nn.BatchNorm2d(conv_channels[3]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())
            else:
                self.conv4.add_module(f"conv4-{i}",
                                      nn.Conv2d(conv_channels[3], conv_channels[3], (3, 3), padding=(1, 1))
                                      )
                self.conv4.add_module(f"bn4-{i}", nn.BatchNorm2d(conv_channels[3]))
                self.conv1.add_module(f"relu-{i}", nn.ReLU())

        self.max_pooling = nn.MaxPool2d((2, 2), (2, 2))

    def forward(self, inp):
        conv1 = self.conv1(inp)  # H, W
        down1 = self.max_pooling(conv1)
        conv2 = self.conv2(down1)  # H/2, W/2
        down2 = self.max_pooling(conv2)
        conv3 = self.conv3(down2)  # H/4, W/4
        down3 = self.max_pooling(conv3)
        conv4 = self.conv4(down3)  # H/8, W/8
        return conv1, conv2, conv3, conv4


class FPNDecoder(nn.Module):

    def __init__(self, conv_channels):
        super(FPNDecoder, self).__init__()
        self.up_conv3 = nn.ConvTranspose2d(conv_channels[3], conv_channels[2], (2, 2), (2, 2))
        self.bn_up3 = nn.BatchNorm2d(conv_channels[2])
        self.up_conv2 = nn.ConvTranspose2d(conv_channels[2], conv_channels[1], (2, 2), (2, 2))
        self.bn_up2 = nn.BatchNorm2d(conv_channels[1])
        self.up_conv1 = nn.ConvTranspose2d(conv_channels[1], conv_channels[0], (2, 2), (2, 2))
        self.bn_up1 = nn.BatchNorm2d(conv_channels[0])

        # pyramid
        self.pyramid_fusion_3 = nn.Conv2d(conv_channels[2] * 2, conv_channels[2], (3, 3), padding=(1, 1))
        self.pyramid_bn3 = nn.BatchNorm2d(conv_channels[2])
        self.pyramid_fusion_2 = nn.Conv2d(conv_channels[1] * 2, conv_channels[1], (3, 3), padding=(1, 1))
        self.pyramid_bn2 = nn.BatchNorm2d(conv_channels[1])
        self.pyramid_fusion_1 = nn.Conv2d(conv_channels[0] * 2, conv_channels[0], (3, 3), padding=(1, 1))
        self.pyramid_bn1 = nn.BatchNorm2d(conv_channels[0])

    def forward(self, conv1, conv2, conv3, conv4):
        up3 = self.bn_up3(self.up_conv3(conv4))  # H/4, W/4
        cat3 = torch.cat([conv3, up3], dim=1)
        fusion_3 = self.pyramid_bn3(self.pyramid_fusion_3(cat3))

        up2 = self.bn_up2(self.up_conv2(fusion_3))  # H/2, W/2
        cat2 = torch.cat([conv2, up2], dim=1)
        fusion_2 = self.pyramid_bn2(self.pyramid_fusion_2(cat2))

        up1 = self.bn_up1(self.up_conv1(fusion_2))  # H, W
        cat1 = torch.cat([conv1, up1], dim=1)
        fusion_1 = self.pyramid_bn1(self.pyramid_fusion_1(cat1))
        return fusion_1


class BEVExtractor(nn.Module):

    def __init__(self, in_channel, conv_channels, repeats):
        super(BEVExtractor, self).__init__()
        # encoder
        self.encoder = VggEncoder(in_channel, conv_channels, repeats)
        # decoder
        self.decoder = FPNDecoder(conv_channels)

    def forward(self, inp):
        conv1, conv2, conv3, conv4 = self.encoder(inp)
        output = self.decoder(conv1, conv2, conv3, conv4)

        return output