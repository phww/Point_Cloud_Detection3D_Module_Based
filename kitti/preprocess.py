#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/26 下午3:06
# @Author : PH
# @Version：V 0.1
# @File : preprocess.py
# @desc :
import pickle
import numpy as np
from pathlib import Path
from kitti.io.kitti_io import KittiIo
from basic.dataset.preprocess.preprocessor import Preprocessor
from basic.box_np_ops.anchor2bbox import anchor3d_to_bbox8c
from basic.box_np_ops.bbox_utils import in_hull
from basic.ops.pc_3rd_ops.roiaware_pool3d import roiaware_pool3d_utils


class KittiPreprocessor(Preprocessor):
    def __init__(self, preprocess_cfg_path):
        super(KittiPreprocessor, self).__init__(preprocess_cfg_path)
        self.kitti_reader = None

    def generate_infos_file(self, save_path, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        self.kitti_reader = KittiIo(self.raw_data_path)

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split_type, sample_idx))
            info = {}
            pc = self.kitti_reader.read_velodyne_bin(sample_idx, filter=False)
            pc_info = {'lidar_idx': sample_idx, 'total_num': pc.shape[0], 'num_features': pc.shape[1]}
            info['point_cloud'] = pc_info

            img = self.kitti_reader.read_image(sample_idx)
            image_info = {'image_idx': sample_idx, 'image_shape': np.array(img.shape, np.int32)}
            info['image'] = image_info

            calib = self.kitti_reader.read_calibration(sample_idx)
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            info['calib'] = calib_info

            if has_label:
                obj_list = self.kitti_reader.read_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.points_camera_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    # keep point cloud that in image
                    pts_cam = calib.points_lidar_to_camera(pc[:, 0:3])
                    fov_flag = calib.filter_pc_in_img(pts_cam, info['image']['image_shape'])
                    pts_fov = pc[fov_flag]

                    # keep point cloud that in Delaunay TIN
                    corners_lidar = anchor3d_to_bbox8c(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # Every self.cfg.DUMP_INTERVAL(1000) infos are dumped，In order to prevent insufficient memory
        start = 0
        end = 0
        while end != len(sample_id_list):
            end = len(sample_id_list) if (start + self.cfg.DUMP_INTERVAL) > len(sample_id_list) \
                else (start + self.cfg.DUMP_INTERVAL)
            with futures.ThreadPoolExecutor(self.workers) as executor:
                infos = executor.map(process_single_scene, sample_id_list[start:end])
            start = end
            self.dump_infos(list(infos), save_path)

    def create_groundtruth_database(self, infos_path=None, split_type='train'):
        import torch
        self.kitti_reader = KittiIo(self.raw_data_path)
        database_save_path = Path(self.save_root) / (
            'gt_database' if split_type == 'train' else ('gt_database_%s' % split_type))
        db_info_save_path = Path(self.save_root) / ('db_infos_%s.pkl' % split_type)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.kitti_reader.read_velodyne_bin(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                used_classes = self.cfg.CREATE_GT_DATABASE.USED_CLASSES
                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.save_root))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


if __name__ == '__main__':
    preprocess_cfg_path = '/home/ph/Desktop/PointCloud/utils_my/kitti/cfg/preprocess_cfg.yaml'
    preprocessor = KittiPreprocessor(preprocess_cfg_path)
    preprocessor.create_infos()
