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
from basic.dataset.dataset import DatasetTemplate
import kitti.io.kitti_io
from basic.utils.run.template import TemplateModel
from basic import model
from basic.utils.config_utils import cfg_from_yaml_file
from kitti.kitti_dataset import get_dataloader
from basic.utils.vis_utils import visualize_one_pc_frame
from basic.utils.common_utils import put_data_to_gpu

class Trainer(TemplateModel):

    def __init__(self, model_list, opt_list=None, lr_scheduler=None,
                 lr_scheduler_type=None, loss_fn=None, train_loader=None,
                 test_loader=None,
                 ):
        super().__init__()
        self.model_list = model_list  # 模型的list
        self.optimizer_list = opt_list  # 优化器的list
        self.criterion = loss_fn
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 下面的可以不设定
        self.lr_scheduler_list = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        # tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, 'runs'))  # 推荐设定
        # 训练时print的间隔
        self.log_per_step = 5  # 推荐按数据集大小设定

    def loss_per_batch(self, batch_dict):
        for model in self.model_list:
            model.train()
        batch_dict = put_data_to_gpu(batch_dict)
        loss_dict = batch_dict
        for cur_model in self.model_list:
            loss_dict = cur_model(loss_dict)
        return loss_dict

    def eval_scores_per_batch(self, batch_dict):
        for model in self.model_list:
            model.eval()
        batch_dict = put_data_to_gpu(batch_dict)
        pred_dict = batch_dict
        for cur_model in self.model_list:
            pred_dict_list = cur_model(pred_dict)
        scores = self.metric(pred_dict_list)
        return scores

    def metric(self, pred_dict_list):
        self.key_metric = "MAP"
        scores = {}
        for pred_dict in pred_dict_list:
            eval_dict = pred_dict['eval_dict']
            if eval_dict.get('MAP', None) is not None:
                scores['MAP'] = scores['MAP'] + eval_dict['MAP'] if scores.get('MAP') else eval_dict['MAP']
            if eval_dict.get('AP', None) is not None:
                scores['AP'] = scores['AP'] + eval_dict['AP'] \
                    if scores.get('AP', False) else eval_dict['AP']
            # if eval_dict.get('Recall', None) is not None:
            #     scores['Recall'] = scores['Recall'] + eval_dict['Recall'] \
            #         if scores.get('Recall', False) else eval_dict['Recall']
            # if eval_dict.get('Precision', None) is not None:
            #     scores['Precision'] = scores['Precision'] + eval_dict['Precision'] \
            #         if scores.get('Precision', False) else eval_dict['Precision']
        for key, value in scores.items():
            scores[key] /= len(pred_dict_list)
        return scores

    def inference(self, batch_dict):
        batch_dict = put_data_to_gpu(batch_dict)
        for cur_model in self.model_list:
            cur_model.eval()
        for cur_model in self.model_list:
            batch_dict = cur_model(batch_dict)
        return batch_dict


def train(top_cfg, state_path=None):
    if isinstance(top_cfg, str):
        top_cfg = cfg_from_yaml_file(top_cfg, merge_subconfig=False)
    dataset_cfg_path = Path(top_cfg.DATASET_CONFIG.CONFIG_PATH)
    model_cfg = top_cfg.MODEL
    train_cfg = top_cfg.TRAIN_CONFIG
    # dataset
    train_loader = get_dataloader(data_cfg_path=dataset_cfg_path, class_name_list=train_cfg.CLASS_NAMES,
                                  batch_size=train_cfg.BATCH, training=True
                                  )
    test_loader = get_dataloader(data_cfg_path=dataset_cfg_path, class_name_list=train_cfg.CLASS_NAMES,
                                 batch_size=train_cfg.BATCH, training=False
                                 )
    # model
    model_obj = model.all[model_cfg.NAME](top_cfg)

    # optimizer
    optim_cfg = train_cfg.OPTIMIZER
    opt = torch.optim.AdamW(model_obj.parameters(),
                            lr=optim_cfg.LR,
                            betas=optim_cfg.BETAS,
                            weight_decay=optim_cfg.WEIGHT_DECAY,
                            amsgrad=optim_cfg.AMSGRAD
                            )
    # if optim_cfg.WARM_UP:
    #     warm_up_scheduler = torch.optim.lr_scheduler.StepLR(opt,1,gamma=)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=0, last_epoch=-1)
    # torch.optim.lr_scheduler.
    # loss
    loss_fn = model_obj.get_training_loss

    # Trainer
    trainer = Trainer([model_obj], [opt], None, 'loss', loss_fn, train_loader, test_loader)
    # trainer = Trainer([model_obj], [opt], [scheduler], 'annealing', loss_fn, train_loader, test_loader)
    trainer.check_init()
    epochs = train_cfg.EPOCHS
    for epoch in range(epochs):
        if state_path is not None:
            trainer.load_state(state_path)
        trainer.train_loop(clip_grad=train_cfg.GRAD_NORM_CLIP)
        trainer.eval_loop(save_per_epochs=train_cfg.SAVE_STATE)
        trainer.update_lr_scheduler()
        trainer.epoch += 1
    trainer.print_best_metrics()
        # trainer.epoch += 1
        # trainer.save_state(os.path.join(trainer.ckpt_dir, f"epoch{epoch}.pkl"), False)
        # if (epoch + 1) % train_cfg.SAVE_STATE == 0:
        #     trainer.save_model(os.path.join(trainer.ckpt_dir, f"epoch{epoch}.pkl"))
        # trainer.save_state(os.path.join(trainer.ckpt_dir, f"epoch{epoch}.pkl"))

    # trainer.eval_loop()


class Predictor:

    def __init__(self, detect_obj_type, model_path=None, data_cfg_path=None, batch_size=None, state_path=None, model_cfg=None):
        self.model = None
        if state_path is not None and model_cfg is not None:
            if isinstance(model_cfg, str):
                model_cfg = cfg_from_yaml_file(model_cfg, merge_subconfig=False)
            self.model = model.all[model_cfg.MODEL.NAME](model_cfg).cuda()
            state = torch.load(state_path)
            self.model.load_state_dict(state['model0'])
        elif model_path is not None:
            with open(model_path, 'rb') as f:
                self.model = torch.load(f)['model0']
        else:
            raise ValueError("model_path or (state_path + model_cfg) needed")
        self.class_names = detect_obj_type
        self.dataloader = None
        if data_cfg_path is not None:
            assert detect_obj_type is not None
            assert batch_size is not None
            self.dataloader = get_dataloader(data_cfg_path=data_cfg_path,
                                             class_name_list=detect_obj_type,
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
    train_path = "/home/ph/Desktop/PointCloud/utils_my/basic/model/model_cfg/second.yaml"
    state_path = None
    train_cfg = cfg_from_yaml_file(train_path, False)
    train(train_cfg, state_path)
