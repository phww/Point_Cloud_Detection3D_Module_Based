#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/18 上午9:38
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import os.path
from pathlib import Path
import easydict
import torch
from torch.utils.tensorboard import SummaryWriter

import kitti.io.kitti_io
from basic.utils.run.template import TemplateModel
from basic import model
from basic.utils.config_utils import cfg_from_yaml_file
from kitti.kitti_dataset import get_dataloader
from basic.utils.vis_utils import visualize_one_pc_frame
from basic.utils.common_utils import put_data_to_gpu


class Trainer(TemplateModel):

    def __init__(self, model_list, opt_list=None, loss_fn=None, train_loader=None, test_loader=None, writer=None):
        super().__init__()
        self.model_list = model_list  # 模型的list
        self.optimizer_list = opt_list  # 优化器的list
        self.criterion = loss_fn
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 下面的可以不设定
        # tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, 'runs'))  # 推荐设定
        # 训练时print的间隔
        self.log_per_step = 5  # 推荐按数据集大小设定

    def loss_per_batch(self, batch_dict):
        batch_dict = put_data_to_gpu(batch_dict)
        for cur_model in self.model_list:
            loss_dict = cur_model(batch_dict)
        return loss_dict

    def eval_scores_per_batch(self, batch):
        pass

    def inference(self, batch_dict):
        batch_dict = put_data_to_gpu(batch_dict)
        for cur_model in self.model_list:
            cur_model.eval()
        for cur_model in self.model_list:
            batch_dict = cur_model(batch_dict)
        return batch_dict


def train(train_cfg):
    if isinstance(train_cfg, str):
        train_cfg = cfg_from_yaml_file(train_cfg, merge_subconfig=False)
    dataset_cfg_path = Path(train_cfg.DATASET_CONFIG)
    model_cfg = cfg_from_yaml_file(train_cfg.MODEL.CONFIG, merge_subconfig=False)
    # dataset
    train_loader = get_dataloader(data_cfg_path=dataset_cfg_path, class_name_list=train_cfg.CLASS_NAMES,
                                  batch_size=train_cfg.BATCH, training=True
                                  )
    test_loader = get_dataloader(data_cfg_path=dataset_cfg_path, class_name_list=train_cfg.CLASS_NAMES,
                                 batch_size=train_cfg.BATCH, training=False
                                 )
    # model
    data_info = train_loader.dataset.get_data_infos()
    model_obj = model.all[train_cfg.MODEL.NAME](model_cfg, data_info)

    # optimizer
    lr = train_cfg.OPTIMIZER.LR
    opt = torch.optim.Adam(model_obj.parameters(), lr=lr)

    # loss
    loss_fn = model_obj.get_training_loss

    # Trainer
    trainer = Trainer([model_obj], [opt], loss_fn, train_loader, test_loader)
    trainer.check_init()
    # trainer.print_all_member()
    epochs = train_cfg.EPOCHS
    for epoch in range(epochs):
        trainer.train_loop()
        # trainer.save_state(os.path.join(trainer.ckpt_dir, f"epoch{epoch}.pkl"), False)
        trainer.save_model(os.path.join(trainer.ckpt_dir, f"epoch{epoch}.pkl"))

    # trainer.eval_loop()


class Predictor:

    def __init__(self, model_path, class_names, data_cfg_path=None, batch_size=None):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f)['model0']
        self.class_names = class_names
        self.dataloader = None
        if data_cfg_path is not None:
            assert class_names is not None
            assert batch_size is not None
            self.dataloader = get_dataloader(data_cfg_path=data_cfg_path,
                                             class_name_list=class_names,
                                             batch_size=batch_size
                                             )

    def predict_bbox(self, data_dict, eval=True):
        if eval:
            self.model.eval()
        else:
            self.model.train()
        pred_dict = self.model(data_dict)
        return pred_dict

    def predict_bbox_from_loader(self):
        assert self.dataloader is not None, "DataLoader is None"
        batch_data = next(iter(self.dataloader))
        batch_data = put_data_to_gpu(batch_data)
        pred_dict = self.predict_bbox(data_dict=batch_data)
        return pred_dict, batch_data

    def predict_and_visualize(self, data_dict=None):
        pred_dict = None
        if data_dict is None:
            pred_dict, batch_data = self.predict_bbox_from_loader()
        else:
            pred_dict = self.predict_bbox(data_dict)
            batch_data = data_dict
        # ground truth in a batch of frames
        frame_ids = batch_data['frame_id']
        gt_boxes = batch_data['gt_boxes']
        gt_boxes = gt_boxes[..., :7]
        gt_labels = gt_boxes[..., -1]

        # predict bbox in a batch of frames
        pred_bboxes = pred_dict['pred_bbox']
        pred_labels = pred_dict['pred_bbox_labels']
        pred_frame_inds = pred_dict['frame_inds']
        pred_frame_ids = frame_ids[pred_frame_inds] if pred_frame_inds is not None else None
        print("pred_bbox:", pred_bboxes)
        print("pred_labels:", pred_labels)
        print("frame_ids", pred_frame_ids)

        # visualize frame by frame
        for i, frame_id in enumerate(frame_ids):
            pred_frame_bboxes = None
            if pred_bboxes is not None:
                pred_frame_labels = pred_labels[pred_frame_ids == frame_id]
                pred_frame_bboxes = pred_bboxes[pred_frame_ids == frame_id]
                num = pred_frame_labels.shape[0]
                print(f"Frame:{frame_id}. Predict boxes' numbers:{num}")
                if num > 0:
                    for box, label_id in zip(pred_frame_bboxes, pred_frame_labels):
                        print(f"predict bbox:{self.class_names[label_id - 1]}<{box}>")
            else:
                print(f"No predict bbox for this batch of frames")
            visualize_one_pc_frame(pc=batch_data['points'].gather(dim=0, index=i),
                                   pred=pred_frame_bboxes,
                                   gt=gt_boxes[i],
                                   file_reader=kitti.io.kitti_io.KittiIo(root="/home/ph/Dataset/KITTI/testing")
                                   )
            flag = input(f"Continue predict?[{i}\{len(frame_ids)}] (push 'q' to quit)")
            if flag == 'q':
                break


if __name__ == '__main__':
    train_path = "/home/ph/Desktop/PointCloud/utils_my/kitti/cfg/train_cfg.yaml"
    train_cfg = cfg_from_yaml_file(train_path, False)
    train(train_cfg)
