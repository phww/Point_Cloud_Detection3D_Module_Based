#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/7/30 下午1:57
# @Author : PH
# @Version：V 0.1
# @File : vis_utils.py
# @desc :
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import kitti.io.kitti_io
from basic.utils.bbox import box_utils
# from kitti_utils.kitti_io import KittiIo
from wavedata.tools.obj_detection import obj_utils


def set_plot_limits(axes, shape):
    # Set the plot limits to the size of the image, y is inverted
    axes.set_xlim(0, shape[1])
    axes.set_ylim(shape[0], 0)


def scale_to_255(a, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - a.min()) / float(a.max() - a.min())) * 255).astype(dtype)


def convert_bev_point2img(pts, res=0.1, cam_off=None, coordinate='Point', flit_extents=None):
    assert coordinate in ["Camera", "Point", "Pixel"]
    # 将坐标原点设置为(0,0,0)最小值，且保证所有点的坐标全正
    x_points = pts[:, 0] - pts[:, 0].min()
    y_points = pts[:, 1] - pts[:, 1].min()
    z_points = pts[:, 2] - pts[:, 2].min()

    # 在相机坐标系下，bev是xz平面的投影。高度为y
    if coordinate == 'Camera':
        u_coord = (x_points / res).astype(np.int32)
        v_coord = (z_points / res).astype(np.int32)
        high = y_points
        y_points = z_points
    # 在点云坐标系下，bev是xy平面的投影。高度为z
    elif coordinate == 'Point':
        u_coord = (y_points / res).astype(np.int32)
        v_coord = (x_points / res).astype(np.int32)
        high = z_points
        y_points = y_points
    # 在像素、图像坐标系下，bev也是xz平面的投影。但是x与y相对于相机的主点偏移了u_0和v_0
    elif coordinate == 'Pixel':
        assert cam_off is not None, 'The coordinate offset of camera and pixel is None'
        u_0 = cam_off[0]
        v_0 = cam_off[1]
        u_coord = ((x_points - u_0) / res).astype(np.int32)
        v_coord = (z_points / res).astype(np.int32)
        high = y_points - v_0
        y_points = z_points

    # 手动筛选flit_extents[x_min, x_max, y_min, y_max, z_min, z_max]内的点
    # if flit_extents is not None:
    #     x_filt = np.logical_and((x_points > flit_extents[0]), (x_points < flit_extents[1]))
    #     y_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    #     filter = np.logical_and(f_filt, s_filt)
    #     extent_mask =

    # 估算图片大小
    img_w = 1 + int(np.ceil((x_points.max() - x_points.min()) / res))
    img_h = 1 + int(np.ceil((y_points.max() - y_points.min()) / res))

    # 筛选输出图像尺寸内的像素点
    mask = np.where((u_coord < img_w) & (v_coord < img_h))
    u_coord = u_coord[mask]
    v_coord = v_coord[mask]

    # 按照v_coord, u_coord与pixel_values填充图像的对应像素
    img = np.zeros((img_h, img_w), np.int8)
    pixel_values = scale_to_255(high[mask])
    img[v_coord, u_coord] = pixel_values
    return img


def visualize_one_pc_frame(pc, pred, gt, file_reader=None):
    vis_window = VisualWindow(mode='3d', file_reader=file_reader)
    vis_window.draw_point_cloud(pc=pc)
    if pred is not None:
        vis_window.draw_boxes3d(boxes=pred, format='corner', c='g')
    if gt is not None:
        vis_window.draw_boxes3d(boxes=gt, format='corner', c='r')


class VisualWindow:

    def __init__(self, mode, img_shape=None, file_reader=None):
        super(VisualWindow, self).__init__()
        self.fig = plt.figure(figsize=(12, 9))
        self.top_img_axes = None
        self.bot_img_axes = None
        self.axes_3d = None
        self.bev_axes = None
        self.img_shape = img_shape
        assert mode in ['single', 'top_bot', '3d', 'bev']
        self.mode = mode
        self.reader = file_reader
        self.init_window()

    def init_window(self):
        self.fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)
        if self.mode == 'single':
            self.top_img_axes = self.fig.add_subplot(111)
        if self.mode == 'top_bot':
            self.top_img_axes = self.fig.add_subplot(211)
            self.bot_img_axes = self.fig.add_subplot(212)
        if self.mode == '3d':
            self.axes_3d = self.fig.add_subplot(111, projection='3d')
        if self.mode == 'bev':
            self.bev_axes = self.fig.add_subplot(111)

        if self.top_img_axes is not None:
            set_plot_limits(self.top_img_axes, self.img_shape)
            self.top_img_axes.set_axis_off()
        if self.bot_img_axes is not None:
            set_plot_limits(self.bot_img_axes, self.img_shape)
            self.bot_img_axes.set_axis_off()
        if self.axes_3d is not None:
            self.axes_3d.set_axis_off()
        if self.bev_axes is not None:
            # pass
            # self.bev_axes.invert_yaxis()
            self.bev_axes.set_axis_off()

    def show(self):
        self.fig.canvas.draw()
        plt.pause(0.3)
        input()

    def draw_img(self, img):
        if self.mode == 'single':
            self.top_img_axes.imshow(img)
        if self.mode == 'top_bot':
            self.top_img_axes.imshow(img)
            self.bot_img_axes.imshow(img)

    def draw_point_cloud(self, pc):
        assert self.axes_3d is not None
        if isinstance(pc, str):
            assert self.reader is not None
            pc = self.reader.read_velodyne_bin(pc)

        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        self.axes_3d.scatter(x, y, z, s=0.4, c=x, cmap="nipy_spectral")

    def draw_bev(self, inp, is_img=True, pc_in_point_coord=True, depth_range=None):
        assert self.bev_axes is not None
        if not is_img:
            if not pc_in_point_coord:
                x = inp[:, 0]
                y = inp[:, 2]
                depth = inp[:, 1]
            else:
                x = inp[:, 0]
                y = inp[:, 1]
                depth = inp[:, 2]
            if depth_range is not None:
                depth = np.clip(depth, depth_range[0], depth_range[1])
            depth = scale_to_255(depth)
            self.bev_axes.scatter(x, y, c=depth, cmap="nipy_spectral", s=0.4, vmin=0, vmax=255)
        else:
            self.bev_axes.imshow(inp, cmap="nipy_spectral", vmin=0, vmax=255)

    def draw_boxes2d_on_img(self, boxes):
        for box in boxes:
            self.draw_single_box2d_on_img(box)

    def draw_single_box2d_on_img(self, box2d):
        rect = patches.Rectangle((box2d[0], box2d[1]), box2d[2] - box2d[0], box2d[3] - box2d[1],
                                 linewidth=1, edgecolor='r', facecolor='none'
                                 )
        self.top_img_axes.add_patch(rect)

    def draw_single_box3d(self, box3d, format='center', c='g'):
        assert self.axes_3d is not None
        if format == 'center':
            plot_3d_cube_center(box3d[:3], box3d[3:6], self.axes_3d, c)
        if format == 'corner':
            plot_3d_cube_corners(box3d, self.axes_3d, c)

    def draw_boxes3d(self, boxes, format='center', c='g'):
        if format == 'corner':
            if boxes.shape[0] != 8:
                boxes = box_utils.boxes_to_corners_3d(boxes)
        for box in boxes:
            self.draw_single_box3d(box, format, c)

    def draw_point_cloud_on_img(self, pc, calib):
        assert self.bot_img_axes is not None
        xy, depth = calib.points_lidar_to_pixel(pc)
        self.bot_img_axes.scatter(xy[:, 0], xy[:, 1], c=depth, s=0.3, cmap='nipy_spectral')


def draw_box_2d(ax, obj, test_mode=False, color_tm='g'):
    """Draws the 2D boxes given the subplot and the object properties

    Keyword arguments:
    :param ax -- subplot handle
    :param obj -- object file to draw bounding bbox
    """

    if not test_mode:
        # define colors
        color_table = ["#00cc00", 'y', 'r', 'w']
        trun_style = ['solid', 'dashed']

        if obj.type != 'DontCare':
            # draw the boxes
            trc = int(obj.truncation > 0.1)
            rect = patches.Rectangle((obj.x1, obj.y1),
                                     obj.x2 - obj.x1,
                                     obj.y2 - obj.y1,
                                     linewidth=2,
                                     edgecolor=color_table[int(obj.occlusion)],
                                     linestyle=trun_style[trc],
                                     facecolor='none'
                                     )

            # draw the labels
            label = "%s\n%1.1f rad" % (obj.type, obj.alpha)
            x = (obj.x1 + obj.x2) / 2
            y = obj.y1
            ax.text(x,
                    y,
                    label,
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    color=color_table[int(obj.occlusion)],
                    fontsize=8,
                    backgroundcolor='k',
                    fontweight='bold'
                    )

        else:
            # Create a rectangle patch
            rect = patches.Rectangle((obj.x1, obj.y1),
                                     obj.x2 - obj.x1,
                                     obj.y2 - obj.y1,
                                     linewidth=2,
                                     edgecolor='c',
                                     facecolor='none'
                                     )

        # Add the patch to the Axes
        ax.add_patch(rect)
    else:
        # we are in test mode, so customize the boxes differently
        # draw the boxes
        # we also don't care about labels here
        rect = patches.Rectangle((obj.x1, obj.y1),
                                 obj.x2 - obj.x1,
                                 obj.y2 - obj.y1,
                                 linewidth=2,
                                 edgecolor=color_tm,
                                 facecolor='none'
                                 )
        # Add the patch to the Axes
        ax.add_patch(rect)


def draw_box_3d(ax, obj, p, show_orientation=True,
                color_table=None, line_width=3, double_line=True,
                box_color=None
                ):
    """Draws the 3D boxes given the subplot, object label,
    and frame transformation matrix

    :param ax: subplot handle
    :param obj: object file to draw bounding bbox
    :param p:stereo frame transformation matrix

    :param show_orientation: optional, draw a line showing orientaion
    :param color_table: optional, a custom table for coloring the boxes,
        should have 4 values to match the 4 truncation values. This color
        scheme is used to display boxes colored based on difficulty.
    :param line_width: optional, custom line width to draw the bbox
    :param double_line: optional, overlays a thinner line inside the bbox lines
    :param box_color: optional, use a custom color for bbox (instead of
        the default color_table.
    """

    corners3d = obj_utils.compute_box_corners_3d(obj)
    corners, face_idx = obj_utils.project_box3d_to_image(corners3d, p)

    # define colors
    if color_table:
        if len(color_table) != 4:
            raise ValueError('Invalid color table length, must be 4')
    else:
        color_table = ["#00cc00", 'y', 'r', 'w']

    trun_style = ['solid', 'dashed']
    trc = int(obj.truncation > 0.1)

    if len(corners) > 0:
        for i in range(4):
            x = np.append(corners[0, face_idx[i,]],
                          corners[0, face_idx[i, 0]]
                          )
            y = np.append(corners[1, face_idx[i,]],
                          corners[1, face_idx[i, 0]]
                          )

            # Draw the boxes
            if box_color is None:
                box_color = color_table[int(obj.occlusion)]

            ax.plot(x, y, linewidth=line_width,
                    color=box_color,
                    linestyle=trun_style[trc]
                    )

            # Draw a thinner second line inside
            if double_line:
                ax.plot(x, y, linewidth=line_width / 3.0, color='b')

    if show_orientation:
        # Compute orientation 3D
        orientation = obj_utils.compute_orientation_3d(obj, p)

        if orientation is not None:
            x = np.append(orientation[0,], orientation[0,])
            y = np.append(orientation[1,], orientation[1,])

            # draw the boxes
            ax.plot(x, y, linewidth=4, color='w')
            ax.plot(x, y, linewidth=2, color='k')


def plot_3d_cube_corners(corners, ax, c='lime'):
    """Plots 3D cube

    Arguments:
        corners: Bounding bbox corners
        ax: graphics handler
    """

    # Draw each line of the cube
    p1 = corners[0]
    p2 = corners[1]
    p3 = corners[2]
    p4 = corners[3]

    p5 = corners[4]
    p6 = corners[5]
    p7 = corners[6]
    p8 = corners[7]

    #############################
    # Bottom Face
    #############################
    ax.plot([p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            c=c
            )

    ax.plot([p2[0], p3[0]],
            [p2[1], p3[1]],
            [p2[2], p3[2]],
            c=c
            )

    ax.plot([p3[0], p4[0]],
            [p3[1], p4[1]],
            [p3[2], p4[2]],
            c=c
            )

    ax.plot([p4[0], p1[0]],
            [p4[1], p1[1]],
            [p4[2], p1[2]],
            c=c
            )

    #############################
    # Top Face
    #############################
    ax.plot([p5[0], p6[0]],
            [p5[1], p6[1]],
            [p5[2], p6[2]],
            c=c
            )

    ax.plot([p6[0], p7[0]],
            [p6[1], p7[1]],
            [p6[2], p7[2]],
            c=c
            )

    ax.plot([p7[0], p8[0]],
            [p7[1], p8[1]],
            [p7[2], p8[2]],
            c=c
            )

    ax.plot([p8[0], p5[0]],
            [p8[1], p5[1]],
            [p8[2], p5[2]],
            c=c
            )

    #############################
    # Front-Back Face
    #############################
    ax.plot([p5[0], p8[0]],
            [p5[1], p8[1]],
            [p5[2], p8[2]],
            c=c
            )

    ax.plot([p8[0], p4[0]],
            [p8[1], p4[1]],
            [p8[2], p4[2]],
            c=c
            )

    ax.plot([p4[0], p1[0]],
            [p4[1], p1[1]],
            [p4[2], p1[2]],
            c=c
            )

    ax.plot([p1[0], p5[0]],
            [p1[1], p5[1]],
            [p1[2], p5[2]],
            c=c
            )

    #############################
    # Front Face
    #############################
    ax.plot([p2[0], p3[0]],
            [p2[1], p3[1]],
            [p2[2], p3[2]],
            c=c
            )

    ax.plot([p3[0], p7[0]],
            [p3[1], p7[1]],
            [p3[2], p7[2]],
            c=c
            )

    ax.plot([p7[0], p6[0]],
            [p7[1], p6[1]],
            [p7[2], p6[2]],
            c=c
            )

    ax.plot([p6[0], p2[0]],
            [p6[1], p2[1]],
            [p6[2], p2[2]],
            c=c
            )


def plot_3d_cube_center(center, size, ax, c):
    """
       Create a data array for cuboid plotting.


       ============= ================================================
       Argument      Description
       ============= ================================================
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)
   """
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    import numpy as np

    ox, oy, oz = center
    l, h, w = size

    x = np.linspace(ox - l / 2, ox + l / 2, num=10)
    y = np.linspace(oy - w / 2, oy + w / 2, num=10)
    z = np.linspace(oz - h / 2, oz + h / 2, num=10)
    x1, z1 = np.meshgrid(x, z)
    y11 = np.ones_like(x1) * (oy - w / 2)
    y12 = np.ones_like(x1) * (oy + w / 2)
    x2, y2 = np.meshgrid(x, y)
    z21 = np.ones_like(x2) * (oz - h / 2)
    z22 = np.ones_like(x2) * (oz + h / 2)
    y3, z3 = np.meshgrid(y, z)
    x31 = np.ones_like(y3) * (ox - l / 2)
    x32 = np.ones_like(y3) * (ox + l / 2)

    # outside surface
    ax.plot_wireframe(x1, y11, z1, color=c, rstride=10, cstride=10, alpha=0.6)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color=c, rstride=10, cstride=10, alpha=0.6)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color=c, rstride=10, cstride=10, alpha=0.6)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color=c, rstride=10, cstride=10, alpha=0.6)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color=c, rstride=10, cstride=10, alpha=0.6)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color=c, rstride=10, cstride=10, alpha=0.6)
    # ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    # ax.set_ylabel('Y')
    # ax.set_ylim(0, 100)
    # ax.set_zlabel('Z')
    # ax.set_zlim(-10, 3)
    # fig.show()
    # plt.pause(0.001)


def cv2_show_image(window_name, image,
                   size_wh=None, location_xy=None
                   ):
    """ Helper function for specifying window size and location when
        displaying images with cv2

    :param window_name:
    :param image:
    :param size_wh:
    :param location_xy:
    """
    if size_wh is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, *size_wh)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    if location_xy is not None:
        cv2.moveWindow(window_name, *location_xy)

    cv2.imshow(window_name, image)


def plot_2d_rect_center(center, ax):
    x1 = center[0]
    z1 = center[1]
    l = center[2]
    w = center[3]
    r = center[4]
    rect = patches.Rectangle((x1, z1), l, w, r, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


def fill_pixel(xy, xy_extent, fill_src, fill_extent, resolution=0.1, filt_points=True):
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = xy[:, 0]
    y_points = xy[:, 1]
    z_points = fill_src

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates

    if filt_points:
        f_filt = np.logical_and((x_points > xy_extent[0][0]), (x_points < xy_extent[0][1]))
        s_filt = np.logical_and((y_points > xy_extent[1][0]), (y_points < xy_extent[1][1]))
        filter = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filter).flatten()
    # else:
    #     indices = np.arange(len(points))

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_points / resolution).astype(np.int32)
    y_img = (y_points / resolution).astype(np.int32)

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift

    if filt_points:
        x_img -= int(np.floor(xy_extent[0][0] / resolution))
        y_img -= int(np.ceil(xy_extent[1][1] / resolution))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=fill_extent[0],
                           a_max=fill_extent[1]
                           )

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    fill_value = (((z_points - fill_src.min()) / float(fill_src.max() - fill_src.min())) * 255).astype(np.uint8)

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int(np.ceil((x_points.max() - x_points.min()) / resolution))
    y_max = 1 + int(np.ceil((y_points.max() - y_points.min()) / resolution))
    if filt_points:
        x_max = 1 + int((xy_extent[0][1] - xy_extent[0][0]) / resolution)
        y_max = 1 + int((xy_extent[1][1] - xy_extent[1][0]) / resolution)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im
