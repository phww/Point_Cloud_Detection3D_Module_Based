#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/3 上午10:12
# @Author : PH
# @Version：V 0.1
# @File : kitti_io.py
# @desc :
import csv
import numpy as np
import os
from PIL import Image
from kitti.io.calib_utils import Calibration
from basic.datatype.object3d import Object3D
from pathlib import Path

class Annotations:
    def __init__(self):
        self.type = []
        self.truncated = []
        self.occluded = []
        self.alpha = []
        self.bbox = []
        self.dimensions = []
        self.location = []
        self.dis_to_cam = []
        self.rotation_y = []
        self.score = 0

    def get_anchors(self):
        anchor_gt = np.concatenate([self.location, self.dimensions, self.rotation_y], axis=-1)
        return anchor_gt


class KittiIo:
    def __init__(self, root):
        self.root = root if isinstance(root, Path) else Path(root)

    def read_velodyne_bin(self, idx, filter=False):
        path = self.root / 'velodyne' / f"{idx}.bin"
        assert path.exists()
        data = np.fromfile(path, dtype=np.float32)
        data = data.reshape(-1, 4)
        if filter:
            data = filter_point_cloud(data)
        return data

    def read_image(self, idx):
        path = self.root / 'image_2' / f"{idx}.png"
        assert path.exists()
        img = np.array(Image.open(path), dtype=np.float32)
        img /= 255.0
        return img

    def read_label(self, idx):
        with open(os.path.join(self.root, 'label_2', f"{idx}.txt")) as f:
            lines = f.readlines()
        objects = [Object3D(line) for line in lines]
        return objects
        #annos = Annotations()
        # content = [line.strip().split(' ') for line in lines]
        # # dropout DontCare
        # object_mask = np.array([idx for idx, obj in enumerate(content) if obj[0] != 'DontCare'])
        # objects = content[object_mask]
        #
        # num_objects = len(objects)
        # annos.type = np.array([x[0] for x in objects])
        # num_gt = len(annos.type)
        # annos.truncated = np.array([float(x[1]) for x in objects])
        # annos.occluded = np.array([int(x[2]) for x in objects])
        # annos.alpha = np.array([float(x[3]) for x in objects])
        # annos.bbox = np.array([[float(info) for info in x[4:8]] for x in objects]).reshape(-1, 4)
        # # dimensions will convert hwl format to standard lhw(camera) format.
        # annos.dimensions = np.array(
        #     [[float(info) for info in x[8:11]] for x in objects]).reshape(
        #     -1, 3)[:, [1, 0, 2]]
        # annos.location = np.array(
        #     [[float(info) for info in x[11:14]] for x in objects]).reshape(-1, 3)
        # annos.dis_to_cam = np.linalg.norm(annos.location)
        # annos.rotation_y = np.array([float(x[14]) for x in content]).reshape(-1, 1)
        # annos.score = np.array([float(x[15]) for x in objects])
        # return annos

    def read_calibration(self, idx):
        """Reads in Calibration file from Kitti Dataset."""
        calib_file = self.root / 'calib' / f"{idx}.txt"
        assert calib_file.exists()
        return Calibration(calib_file)

    def read_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root / 'depth_2' / f"{idx}.png"
        assert depth_file.exists()
        depth = np.array(Image.open(depth_file), dtype=np.uint8)
        depth /= 256.0
        return depth

    def get_road_plane(self, idx):
        plane_file = self.root / 'planes' / f"{idx}.txt"
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        pts_img, pts_rect_depth = calib.points_camera_to_pixel(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag


def filter_point_cloud(pc, y_extent=(-40, 40.0), x_extent=(0, 70.8), z_extent=(-3, 3)):
    # FILTER - To return only indices of points within desired cube
    # the extent is based on point cloud coordinate!!
    x_points = pc[:, 0]
    y_points = pc[:, 1]
    z_points = pc[:, 2]
    x_filt = np.logical_and((x_points > x_extent[0]), (x_points < x_extent[1]))
    y_filt = np.logical_and((y_points > y_extent[0]), (y_points < y_extent[1]))
    z_filt = np.logical_and((z_points > z_extent[0]), (z_points < z_extent[1]))
    filter = np.logical_and(x_filt, y_filt)
    filter = np.logical_and(filter, z_filt)
    indices = np.argwhere(filter).flatten()
    return pc[indices]
