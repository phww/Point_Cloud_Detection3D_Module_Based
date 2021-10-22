#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/19 下午2:28
# @Author : PH
# @Version：V 0.1
# @File : test_and_vis.py
# @desc :
import torch
from basic.model.second import SECOND
from pathlib import Path
from kitti.kitti_dataset import get_dataloader
from basic.utils.config_utils import cfg_from_yaml_file


class Inference:

    def __init__(self, model_cfg_path, model_state_path):
        model_cfg = cfg_from_yaml_file(model_cfg_path)  # '../basic/model/model_cfg/second.yaml'

    def predict_bbox_and_show(model_cfg_path, data, device='cpu'):
        model = SECOND(model_cfg, data_infos=None).to(device)
        with open('epoch4.pkl', 'rb') as f:
            state = torch.load(f)
        model.load_state_dict(state['model0'])
        model.eval()
        pred_bbox, pred_bbox_labels, frame_ids = model(data)
