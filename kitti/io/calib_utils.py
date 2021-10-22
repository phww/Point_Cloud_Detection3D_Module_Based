#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/2 下午3:59
# @Author : PH
# @Version：V 0.1
# @File : calib_utils.py
# @desc :
import copy

import numpy as np


def read_calibration(calib_path):
    with open(calib_path) as f:
        lines = f.readlines()
    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P0': P0.reshape(3, 4),
            'P1': P1.reshape(3, 4),
            'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4),
            'Tr_imu2velo': Tr_imu_to_velo.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_path):
        if not isinstance(calib_path, dict):
            calib = read_calibration(calib_path)
        else:
            calib = calib_path
        self.P0 = calib['P0']  # 3 x 4
        self.P1 = calib['P1']  # 3 x 4 right
        self.P2 = calib['P2']  # 3 x 4 left
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
        self.I2V = calib['Tr_imu2velo']  # 3 x 4

        # P2/left camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    @staticmethod
    def homogenize_points(pts):
        """
        Homogenize coordinates pts:(N, 3 or 2) to pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def get_calib_mat(self):
        """
        Converts calibration object to transformation matrices
        Args:
            calib: calibration.Calibration, Calibration object
        Returns
            V2R: (4, 4), Lidar to rectified camera transformation matrix
            P2: (3, 4), Camera projection matrix
        """
        V2C = np.vstack((self.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        R0 = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        V2R = R0 @ V2C
        P2 = self.P2
        return V2R, P2

    def points_camera_to_lidar(self, pts_cam):
        """Coordinate conversion：camera coordinate to lidar coordinate"""

        pts_cam = self.homogenize_points(pts_cam)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1

        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = pts_cam @ np.linalg.inv((R0_ext @ V2C_ext).T)
        return pts_lidar[:, 0:3]

    def points_lidar_to_camera(self, pts_lidar):
        """Coordinate conversion：lidar coordinate to camera coordinate"""
        pts_lidar = self.homogenize_points(pts_lidar)
        camera_points = pts_lidar @ (self.R0 @ self.V2C).T
        return camera_points

    def points_camera_to_pixel(self, pts_cam, poj_mat='P2'):
        """
        Coordinate conversion：Camera coordinate to Pixel coordinate
        use proj_mat project points(N,3) to image(N,2)
        For example,for KITTI use P2 project points to left_image
        """
        pts_cam = self.homogenize_points(pts_cam)
        if poj_mat == 'P0':
            pts_2d = pts_cam @ self.P0.T

        if poj_mat == 'P1':
            pts_2d = pts_cam @ self.P1.T

        if poj_mat == 'P2':
            pts_2d = pts_cam @ self.P2.T
        else:
            raise ValueError
        pts_img = (pts_2d[:, 0:2].T / pts_2d[:, 2]).T  # (N, 2)
        pts_cam_depth = pts_2d[:, 2] - self.P2.T[3, 2]  # depth in camera coord
        return pts_img, pts_cam_depth

    def points_lidar_to_pixel(self, pts_lidar):
        pts_cam = self.points_lidar_to_camera(pts_lidar)
        pts_img, pts_cam_depth = self.points_camera_to_pixel(pts_cam)
        return pts_img, pts_cam_depth

    ################################################################################3
    def points_img_to_camera(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_cam = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_cam

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def boxes3d_lidar_to_camera(self, boxes3d_lidar):
        boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
        xyz_lidar = boxes3d_lidar_copy[:, 0:3]
        l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
        r = boxes3d_lidar_copy[:, 6:7]
        # todo：？？
        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = self.points_lidar_to_camera(xyz_lidar)
        # xyz_cam[:, 1] += h.reshape(-1) / 2
        r = -r - np.pi / 2
        return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

    def boxes3d_camera_to_lidar(self, boxes3d_camera):
        """
        Args:
            boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords

        Returns:
            boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        """
        boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
        xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
        l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

        xyz_lidar = self.points_camera_to_lidar(xyz_camera)
        # todo:??
        xyz_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

    def boxes3d_camera_to_boxes2d_pixel(self, boxes3d, image_shape=None):
        """
        :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        :return:
            box_2d_preds: (N, 4) [x1, y1, x2, y2]
        """
        corners3d = self.boxes3d_to_corners3d_camera(boxes3d)
        pts_img, _ = self.points_camera_to_pixel(corners3d.reshape(-1, 3))
        corners_in_image = pts_img.reshape(-1, 8, 2)

        min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
        max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image

    def boxes3d_to_corners3d_camera(self, boxes3d, bottom_center=True):
        """
        :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
        :param bottom_center: whether y is on the bottom center of object
        :return: corners3d: (N, 8, 3)
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        """
        boxes_num = boxes3d.shape[0]
        l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
        x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
        z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
        if bottom_center:
            y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
            y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
        else:
            y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.],
                                 dtype=np.float32).T

        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros, ones, zeros],
                             [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

        x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

        x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
        y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
        z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

        corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

        return corners.astype(np.float32)

    def filter_pc_in_img(self, pts_cam, img_shape):
        pts_img, pts_cam_depth = self.points_camera_to_pixel(pts_cam)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_cam_depth >= 0)
        return pts_valid_flag
