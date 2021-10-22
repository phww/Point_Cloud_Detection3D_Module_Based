#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/28 下午2:41
# @Author : PH
# @Version：V 0.1
# @File : config_utils.py
# @desc :
import yaml
import easydict


def _merge_config(config, merge_config):
    with open(config[merge_config]['CONFIG_PATH'], 'rb') as f:
        try:
            sub_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            sub_config = yaml.load(f)
        config[merge_config].update(sub_config)
    return config


def cfg_from_yaml_file(cfg_file, merge_subconfig=True):
    with open(cfg_file, 'rb') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
        if merge_subconfig:
            if "DATASET_CONFIG" in config.keys():
                config = _merge_config(config=config, merge_config="DATASET_CONFIG")
            if "PERPRO_CONFIG" in config.keys():
                config = _merge_config(config=config, merge_config="PERPRO_CONFIG")
        config = easydict.EasyDict(config)
    return config
