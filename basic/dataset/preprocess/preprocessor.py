#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/27 上午10:56
# @Author : PH
# @Version：V 0.1
# @File : preprocessor.py
# @desc :
import pickle
from pathlib import Path

import yaml
from easydict import EasyDict


class Preprocessor(object):
    def __init__(self, preprocess_cfg_path):
        with open(preprocess_cfg_path, 'rb') as f:
            preprocess_cfg = EasyDict(yaml.load(f))
        self.cfg = preprocess_cfg
        self.split_type = None
        self.data_root = Path(self.cfg.DATA_ROOT)
        self.raw_data_path = None
        self.save_root = Path(self.cfg.SAVE_ROOT)
        self.sample_id_list = []
        self.workers = self.cfg.WORKER_NUM

    def generate_infos_file(self, save_path, has_label=True, count_inside_pts=True):
        raise NotImplementedError

    def create_groundtruth_database(self, infos_path, split_type='train'):
        raise NotImplementedError

    def create_infos(self):
        train_filename = self.save_root / 'infos_train.pkl'
        val_filename = self.save_root / 'infos_val.pkl'
        trainval_filename = self.save_root / 'infos_train_val.pkl'
        test_filename = self.save_root / 'infos_test.pkl'

        print('---------------Start to generate data infos---------------')
        # train_infos
        self.set_split('train')
        self.generate_infos_file(save_path=train_filename, has_label=True, count_inside_pts=True)
        print('train info file is saved to %s' % train_filename)

        # val_infos
        self.set_split('val')
        self.generate_infos_file(save_path=val_filename, has_label=True, count_inside_pts=True)
        print('val info file is saved to %s' % val_filename)

        # # train + val infos
        # with open(trainval_filename, 'wb') as f:
        #     pickle.dump(infos_train + infos_val, f)
        # print('train & val info  file is saved to %s' % trainval_filename)

        # test infos
        self.set_split('test')
        self.generate_infos_file(save_path=test_filename, has_label=False, count_inside_pts=False)
        print('test info file is saved to %s' % test_filename)

        if self.cfg.CREATE_GT_DATABASE.FLAG:
            print('---------------Start create groundtruth database for data augmentation---------------')
            self.set_split('train')
            self.create_groundtruth_database(infos_path=train_filename, split_type='train')
            print('---------------Data preparation Done---------------')

    def set_split(self, split):
        self.split_type = split
        self.raw_data_path = self.data_root / ('training' if self.split_type != 'test' else 'testing')
        split_dir = self.data_root / 'ImageSets' / (self.split_type + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    @staticmethod
    def dump_infos(infos, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(infos, f)
