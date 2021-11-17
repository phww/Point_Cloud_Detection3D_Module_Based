#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/27 下午2:24
# @Author : PH
# @Version：V 0.1
# @File : build.py.py
# @desc :
import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class PostInstallation(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        # Note: buggy for kornia==0.5.3 and it will be fixed in the next version.
        # Set kornia to 0.5.2 temporarily
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'kornia==0.5.2', '--no-dependencies'])


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources])
    return cuda_ext


setup(
    name='pc_3rd_ops',
    cmdclass={
        'build_ext': BuildExtension,
    },
    ext_modules=[
        make_cuda_ext(
            name='iou3d_nms_cuda',
            module='pc_3rd_ops.iou3d_nms',
            sources=[
                'src/iou3d_cpu.cpp',
                'src/iou3d_nms_api.cpp',
                'src/iou3d_nms.cpp',
                'src/iou3d_nms_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='roiaware_pool3d_cuda',
            module='pc_3rd_ops.roiaware_pool3d',
            sources=[
                'src/roiaware_pool3d.cpp',
                'src/roiaware_pool3d_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='roipoint_pool3d_cuda',
            module='pc_3rd_ops.roipoint_pool3d',
            sources=[
                'src/roipoint_pool3d.cpp',
                'src/roipoint_pool3d_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='pointnet2_stack_cuda',
            module='pc_3rd_ops.pointnet2.pointnet2_stack',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/voxel_query.cpp',
                'src/voxel_query_gpu.cu',
            ],
        ),
        make_cuda_ext(
            name='pointnet2_batch_cuda',
            module='pc_3rd_ops.pointnet2.pointnet2_batch',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',

            ],
        ),
    ],
)
