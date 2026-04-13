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


def _set_actor_color(self, env_handle, actor_handle, color):
    self.gym.set_rigid_body_color(env_handle, actor_handle, 0, gymapi.MESH_VISUAL, color)


def _create_fixed_box_actor(self, env_handle, size_xyz, pos_xyz, name, color, quat=None):
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    asset = self.gym.create_box(self.sim, size_xyz[0], size_xyz[1], size_xyz[2], options)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(pos_xyz[0], pos_xyz[1], pos_xyz[2])
    if quat is not None:
        pose.r = quat
    actor_handle = self.gym.create_actor(env_handle, asset, pose, name)
    _set_actor_color(self, env_handle, actor_handle, color)
    return actor_handle


def _create_disc_marker(self, env_handle, center_xy, radius, line_width, thickness, segments, name_prefix, color):
    count = 0
    angle_step = 2 * math.pi / segments
    arc_length = 2 * radius * math.tan(angle_step / 2)
    for i in range(segments):
        mid_angle = i * angle_step + angle_step / 2
        pos_x = center_xy[0] + radius * math.cos(mid_angle)
        pos_y = center_xy[1] + radius * math.sin(mid_angle)
        rot_z = mid_angle + math.pi / 2
        half_angle = rot_z / 2.0
        quat = gymapi.Quat(0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
        _create_fixed_box_actor(
            self,
            env_handle,
            (arc_length, line_width, thickness),
            (pos_x, pos_y, thickness / 2 + 0.002),
            f"{name_prefix}_{i}",
            color,
            quat=quat,
        )
        count += 1
    return count
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


def create_humanoid_adult_field(self, env_handle, field_cfg):
    """
    Create a RoboCup Humanoid League AdultSize-style field using a moderate number
    of fixed visual actors so the scene remains lightweight enough for GPU PhysX.
    """
    length = float(field_cfg["length"])
    width = float(field_cfg["width"])
    line_width = float(field_cfg["line_width"])
    line_thickness = float(field_cfg.get("line_thickness", 0.015))
    center_circle_radius = float(field_cfg["center_circle_radius"])
    goal_area_length = float(field_cfg["goal_area_length"])
    goal_area_width = float(field_cfg["goal_area_width"])
    penalty_area_length = float(field_cfg["penalty_area_length"])
    penalty_area_width = float(field_cfg["penalty_area_width"])
    penalty_mark_distance = float(field_cfg["penalty_mark_distance"])
    goal_width = float(field_cfg["goal_width"])
    goal_depth = float(field_cfg["goal_depth"])
    goal_post_size = float(field_cfg.get("goal_post_size", 0.10))
    goal_height = float(field_cfg["goal_height"])
    border_strip_width = float(field_cfg.get("border_strip_width", 1.0))
    add_grass = bool(field_cfg.get("enable_grass_strips", False))
    add_ground_patch = bool(field_cfg.get("enable_ground_patch", True))
    add_goals = bool(field_cfg.get("enable_goals", True))

    white = gymapi.Vec3(1.0, 1.0, 1.0)
    goal_left = gymapi.Vec3(0.35, 0.65, 1.0)
    goal_right = gymapi.Vec3(1.0, 0.82, 0.35)
    pitch_green = gymapi.Vec3(0.18, 0.52, 0.18)
    count = 0

    if add_ground_patch:
        _create_fixed_box_actor(
            self,
            env_handle,
            (length + 2 * border_strip_width, width + 2 * border_strip_width, 0.01),
            (0.0, 0.0, 0.005),
            "pitch_ground_patch",
            pitch_green,
        )
        count += 1

    if add_grass:
        count += create_strip_grass(self, env_handle, length=length + 2 * border_strip_width, width=width + 2 * border_strip_width, num_strips=10)

    # Outer boundary.
    count += create_field_boundary_lines(self, env_handle, length=length, width=width, line_width=line_width, thickness=line_thickness)

    # Halfway line.
    _create_fixed_box_actor(
        self,
        env_handle,
        (line_width, width, line_thickness),
        (0.0, 0.0, line_thickness / 2 + 0.002),
        "halfway_line",
        white,
    )
    count += 1

    # Center circle + center mark.
    count += _create_disc_marker(
        self,
        env_handle,
        center_xy=(0.0, 0.0),
        radius=center_circle_radius,
        line_width=line_width,
        thickness=line_thickness,
        segments=36,
        name_prefix="center_circle_arc",
        color=white,
    )
    _create_fixed_box_actor(
        self,
        env_handle,
        (line_width, line_width, line_thickness),
        (0.0, 0.0, line_thickness / 2 + 0.002),
        "center_mark",
        white,
    )
    count += 1

    # Goal area and penalty area rectangles.
    for side_sign in (-1.0, 1.0):
        goal_line_x = side_sign * length / 2
        toward_field = -side_sign

        def _rect_from_goal_line(area_length, area_width, prefix):
            x_inner = goal_line_x + toward_field * area_length
            y_half = area_width / 2
            _create_fixed_box_actor(
                self,
                env_handle,
                (line_width, area_width, line_thickness),
                (x_inner, 0.0, line_thickness / 2 + 0.002),
                f"{prefix}_front_{'left' if side_sign < 0 else 'right'}",
                white,
            )
            _create_fixed_box_actor(
                self,
                env_handle,
                (area_length, line_width, line_thickness),
                ((goal_line_x + x_inner) / 2, y_half, line_thickness / 2 + 0.002),
                f"{prefix}_top_{'left' if side_sign < 0 else 'right'}",
                white,
            )
            _create_fixed_box_actor(
                self,
                env_handle,
                (area_length, line_width, line_thickness),
                ((goal_line_x + x_inner) / 2, -y_half, line_thickness / 2 + 0.002),
                f"{prefix}_bottom_{'left' if side_sign < 0 else 'right'}",
                white,
            )
            return 3

        count += _rect_from_goal_line(goal_area_length, goal_area_width, "goal_area")
        count += _rect_from_goal_line(penalty_area_length, penalty_area_width, "penalty_area")

        penalty_x = goal_line_x + toward_field * penalty_mark_distance
        _create_fixed_box_actor(
            self,
            env_handle,
            (line_width, line_width, line_thickness),
            (penalty_x, 0.0, line_thickness / 2 + 0.002),
            f"penalty_mark_{'left' if side_sign < 0 else 'right'}",
            white,
        )
        count += 1

    # Goal frames.
    if add_goals:
        for side_sign, color in ((-1.0, goal_left), (1.0, goal_right)):
            goal_line_x = side_sign * length / 2
            post_x = goal_line_x + side_sign * goal_post_size / 2
            crossbar_x = goal_line_x + side_sign * goal_depth / 2
            y_half = goal_width / 2
            z_half = goal_height / 2

            _create_fixed_box_actor(
                self,
                env_handle,
                (goal_post_size, goal_post_size, goal_height),
                (post_x, y_half, z_half),
                f"goal_post_top_{'left' if side_sign < 0 else 'right'}",
                color,
            )
            _create_fixed_box_actor(
                self,
                env_handle,
                (goal_post_size, goal_post_size, goal_height),
                (post_x, -y_half, z_half),
                f"goal_post_bottom_{'left' if side_sign < 0 else 'right'}",
                color,
            )
            _create_fixed_box_actor(
                self,
                env_handle,
                (goal_depth, goal_post_size, goal_post_size),
                (crossbar_x, 0.0, goal_height - goal_post_size / 2),
                f"goal_crossbar_{'left' if side_sign < 0 else 'right'}",
                color,
            )
            count += 3

    return count
