import os

from isaacgym import gymtorch, gymapi,gymutil
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch
import math
import numpy as np
def create_strip_grass(self, env_handle, length=105.0, width=68.0, num_strips=10, thickness=0.01):
    """
    创建一片草地，沿长边方向分割成num_strips条草带，颜色交替变化。

    参数:
        env_handle: Isaac Gym环境句柄
        length: 草地长（X轴方向，单位米）
        width: 草地宽（Y轴方向，单位米）
        num_strips: 草带数量（越多越细）
        thickness: 草带厚度（Z轴）
    返回值：
        实体数量（1）
    """

    patch_length = length / num_strips  # 每条草带长度

    grass_asset_options = gymapi.AssetOptions()
    grass_asset_options.fix_base_link = True
    grass_asset_options.disable_gravity = True

    for i in range(num_strips):
        patch_asset = self.gym.create_box(
            self.sim,
            patch_length,
            width,
            thickness,
            grass_asset_options
        )

        # 草带中心位置，X方向依次排开，Y方向居中，Z方向抬高防止Z-fighting
        pos_x = (i + 0.5) * patch_length - length / 2
        pos_y = 0.0
        pos_z = thickness / 2 + 0.001  # 抬高1mm防止闪烁

        patch_pose = gymapi.Transform()
        patch_pose.p = gymapi.Vec3(pos_x, pos_y, pos_z)

        patch_handle = self.gym.create_actor(
            env_handle,
            patch_asset,
            patch_pose,
            f"grass_strip_{i}"
        )

        # 颜色交替，绿度在0.5~0.8之间变化
        base_green = 0.65
        variation = 0.15
        green_val = base_green + variation * ((i % 2) * 2 - 1)  # 奇偶条纹不同绿深浅
        green_val = max(0.4, min(0.8, green_val))
        color = gymapi.Vec3(0.1, green_val, 0.1)

        self.gym.set_rigid_body_color(env_handle, patch_handle, 0, gymapi.MESH_VISUAL, color)
    return 15

def create_field_boundary_lines(self, env_handle, length=105.0, width=68.0, line_width=0.15, thickness=0.015):
    """
    创建足球场四周边线（白色）

    参数:
        env_handle: Isaac Gym环境句柄
        length: 场地长（X轴方向）
        width: 场地宽（Y轴方向）
        line_width: 边线宽度（米）
        thickness: 边线厚度（Z轴方向）
    返回：绘制的线条数量（4）
    """
    line_options = gymapi.AssetOptions()
    line_options.fix_base_link = True
    line_options.disable_gravity = True

    # 边线有四条：长边2条，宽边2条

    # 长边线（两条）
    for sign in [-1, 1]:
        line_asset = self.gym.create_box(
            self.sim,
            length + 2 * line_width,  # 多加两端线宽覆盖角落
            line_width,
            thickness,
            line_options
        )
        line_pose = gymapi.Transform()
        line_pose.p = gymapi.Vec3(0.0, sign * (width / 2 + line_width / 2), thickness / 2 + 0.002)
        line_handle = self.gym.create_actor(env_handle, line_asset, line_pose, f"line_long_{sign}")

        white = gymapi.Vec3(1.0, 1.0, 1.0)
        self.gym.set_rigid_body_color(env_handle, line_handle, 0, gymapi.MESH_VISUAL, white)

    # 宽边线（两条）
    for sign in [-1, 1]:
        line_asset = self.gym.create_box(
            self.sim,
            line_width,
            width + 2 * line_width,
            thickness,
            line_options
        )
        line_pose = gymapi.Transform()
        line_pose.p = gymapi.Vec3(sign * (length / 2 + line_width / 2), 0.0, thickness / 2 + 0.002)
        line_handle = self.gym.create_actor(env_handle, line_asset, line_pose, f"line_width_{sign}")

        white = gymapi.Vec3(1.0, 1.0, 1.0)
        self.gym.set_rigid_body_color(env_handle, line_handle, 0, gymapi.MESH_VISUAL, white)
    return 4

def create_field_auxiliary_lines(self, env_handle, length=105.0, width=68.0, line_width=0.15, thickness=0.015):
    """
    绘制足球场辅助线，包括：
    - 中线 1条
    - 中圈 用多条弧线近似 36条
    - 点球点圆 40条弧线(20条*2个点球点)

    返回绘制的线条总数
    """
    line_options = gymapi.AssetOptions()
    line_options.fix_base_link = True
    line_options.disable_gravity = True

    white = gymapi.Vec3(1.0, 1.0, 1.0)
    count = 0

    # 1. 中线（1条）
    mid_line_asset = self.gym.create_box(
        self.sim,
        line_width,
        width,
        thickness,
        line_options
    )
    mid_line_pose = gymapi.Transform()
    mid_line_pose.p = gymapi.Vec3(0.0, 0.0, thickness / 2 + 0.002)
    mid_line_handle = self.gym.create_actor(env_handle, mid_line_asset, mid_line_pose, "mid_line")
    self.gym.set_rigid_body_color(env_handle, mid_line_handle, 0, gymapi.MESH_VISUAL, white)
    count += 1

    # 2. 中圈弧线（36条）
    circle_radius = 3.05
    num_circle_segments = 36
    angle_step = 2 * math.pi / num_circle_segments
    arc_thickness = line_width
    arc_length = 2 * circle_radius * math.tan(angle_step / 2)
    for i in range(num_circle_segments):
        mid_angle = i * angle_step + angle_step / 2
        pos_x = circle_radius * math.cos(mid_angle)
        pos_y = circle_radius * math.sin(mid_angle)
        rot_z = mid_angle + math.pi / 2

        # 计算绕Z轴旋转的四元数
        half_angle = rot_z / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        quat = gymapi.Quat(0.0, 0.0, sin_half, cos_half)

        arc_asset = self.gym.create_box(self.sim, arc_length, arc_thickness, thickness, line_options)
        arc_pose = gymapi.Transform()
        arc_pose.p = gymapi.Vec3(pos_x, pos_y, thickness / 2 + 0.002)
        arc_pose.r = quat

        arc_handle = self.gym.create_actor(env_handle, arc_asset, arc_pose, f"mid_circle_arc_{i}")
        self.gym.set_rigid_body_color(env_handle, arc_handle, 0, gymapi.MESH_VISUAL, white)
        count += 1

    return count