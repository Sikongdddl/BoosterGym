import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_from_euler_xyz,
)

import torch
import numpy as np
from envs.base_task import BaseTask
from utils.scene import *

class LowLevelController(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.addtional_rigid_num = 0
        self.num_agents = self.cfg["env"].get("num_agents", 1)
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        
    def _create_envs(self):
        # 1. robot asset and ball asset
        booster_asset_cfg = self.cfg["asset"]
        booster_asset_root = os.path.dirname(booster_asset_cfg["file"])
        booster_asset_file = os.path.basename(booster_asset_cfg["file"])

        booster_asset_options = gymapi.AssetOptions()
        booster_asset_options.default_dof_drive_mode = booster_asset_cfg["default_dof_drive_mode"]
        booster_asset_options.collapse_fixed_joints = booster_asset_cfg["collapse_fixed_joints"]
        booster_asset_options.replace_cylinder_with_capsule = booster_asset_cfg["replace_cylinder_with_capsule"]
        booster_asset_options.flip_visual_attachments = booster_asset_cfg["flip_visual_attachments"]
        booster_asset_options.fix_base_link = booster_asset_cfg["fix_base_link"]
        booster_asset_options.density = booster_asset_cfg["density"]
        booster_asset_options.angular_damping = booster_asset_cfg["angular_damping"]
        booster_asset_options.linear_damping = booster_asset_cfg["linear_damping"]
        booster_asset_options.max_angular_velocity = booster_asset_cfg["max_angular_velocity"]
        booster_asset_options.max_linear_velocity = booster_asset_cfg["max_linear_velocity"]
        booster_asset_options.armature = booster_asset_cfg["armature"]
        booster_asset_options.thickness = booster_asset_cfg["thickness"]
        booster_asset_options.disable_gravity = booster_asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, booster_asset_root, booster_asset_file, booster_asset_options)
        
        ball_radius = 0.11
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.density = 80.0
        ball_asset_options.disable_gravity = False
        ball_asset_options.fix_base_link = False
        ball_asset_options.linear_damping = 0.5
        ball_asset_options.angular_damping = 0.10
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, ball_asset_options)

        # === 2. DOF/Body info        
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies_robot = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()
        
        # === 2.1 multi agent DOF/Body info
        self.dofs_per_agent = self.num_dofs
        self.bodies_per_agent = self.num_bodies_robot
        self.total_dofs = self.dofs_per_agent * self.num_agents
        self.total_robot_bodies = self.bodies_per_agent * self.num_agents
        
        # 基于“顺序拼接”的切片（典型多机布局：agent0在[0:N)，agent1在[N:2N) ...）
        self.dof_slices  = [slice(i * self.dofs_per_agent,  (i + 1) * self.dofs_per_agent) for i in range(self.num_agents)]
        self.body_slices = [slice(i * self.bodies_per_agent, (i + 1) * self.bodies_per_agent) for i in range(self.num_agents)]
        self.extra_body_indices = {}

        # === 3. DOF stiffness/damping/friction
        self.dof_stiffness = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")

        # === 4. body names & index
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, booster_asset_cfg["base_name"])

        # === 5. shape indices
        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(booster_asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(booster_asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, booster_asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        # === 6. base init state
        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)

        # === 7. envs & actors  ----------------------------------------------------
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []

        # single env and multi agent
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        self.envs.append(env_handle)

        spacing = float(self.cfg.get("env", {}).get("spawn", {}).get("spacing", 1.5))

        for i in range(self.num_agents):
            start_pose = gymapi.Transform()
            # 你之前用的是沿 Y 轴排布；保持一致（如需改为 X 轴，改成 x=i*spacing, y=0）
            start_pose.p = gymapi.Vec3(0.0, i * spacing, self.base_init_state[2].item())
            actor_i = self.gym.create_actor(
                env_handle, robot_asset, start_pose,
                booster_asset_cfg["name"] + f"_{i}",
                0, booster_asset_cfg["self_collisions"], 0
            )
            self.actor_handles.append(actor_i)

            # <<< PATCH3: 对“每个”actor设置刚体属性（你原先只对单个 actor 设过一次）
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_i)
            for j in range(self.num_bodies_robot):
                if j == self.base_indice:
                    body_props[j].com.x = 0.0
                    body_props[j].com.y = 0.0
                    body_props[j].com.z = 0.0
                    body_props[j].mass  = 1.0
                else:
                    body_props[j].com.x = 0.0
                    body_props[j].com.y = 0.0
                    body_props[j].com.z = 0.0
                    body_props[j].mass  = 1.0
                body_props[j].invMass = 1.0 / max(1e-9, body_props[j].mass)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_i, body_props, recomputeInertia=True)

            # 取消随机脚底属性；对“每个”actor逐一设置
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_i)
            for idx in self.foot_shape_indices:
                shape_props[idx].friction = 1.05
                shape_props[idx].compliance = 0.3
                shape_props[idx].restitution = 0.1
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_i, shape_props)

            # 开启 DOF 力传感器（对每个 actor）
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_i)

        # 8 add reachable ball ------------------------------------------------------
        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(3.0, 0.0, ball_radius + 0.02)
        self.ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "SoccerBall", 0, 0)
        self.addtional_rigid_num += 1  # （原拼写保留）

        # 8.1 set ball properties
        ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, self.ball_handle)
        for sp in ball_shape_props:
            sp.friction = 0.45
            if hasattr(sp, "rolling_friction"):
                sp.rolling_friction = 0.05
            if hasattr(sp, "torsion_friction"):
                sp.torsion_friction = 0.02
            sp.restitution = 0.05
        self.gym.set_actor_rigid_shape_properties(env_handle, self.ball_handle, ball_shape_props)

        # 9 add other assets --------------------------------------------------------
        # 这些函数内部会各自创建额外的 actor/刚体并返回计数；保留你的原逻辑
        self.addtional_rigid_num += create_strip_grass(self, env_handle, length=40.0, width=25.0, num_strips=15)
        self.addtional_rigid_num += create_field_boundary_lines(self, env_handle, length=40.0, width=25.0, line_width=0.15)
        self.addtional_rigid_num += create_field_auxiliary_lines(self, env_handle, length=40, width=25)

        # === 10. 索引发现（Discovery Pass）----------------------------------------
        # 目的：不再“猜”球/草坪等的索引；显式记录每个 robot 的刚体/DOF 全局索引表，
        #       以及额外 actor（球等）的刚体全局索引。后续 reset/obs/施力都基于这张真值表。

        # 小工具：若索引严格连续，则压成 slice；否则返回 None（走稀疏索引路径）
        def _compress_to_slice(idx: torch.Tensor):
            if idx.numel() <= 1:
                return slice(int(idx.item()), int(idx.item()) + 1) if idx.numel() == 1 else None
            dif = idx[1:] - idx[:-1]
            if torch.all(dif == 1):
                return slice(int(idx[0].item()), int(idx[-1].item()) + 1)
            return None

        self.rb_indices_per_agent = []
        self.dof_indices_per_agent = []
        self.body_slices = []
        self.dof_slices  = []  # 覆盖前面的“顺序拼接切片”，以“真实发现”为准

        for actor_i in self.actor_handles:
            # 刚体全局索引（ENV 域）
            rb_list = []
            for j in range(self.bodies_per_agent):
                g = self.gym.get_actor_rigid_body_index(env_handle, actor_i, j, gymapi.DOMAIN_ENV)
                rb_list.append(g)
            rb_idx = torch.as_tensor(rb_list, device=self.device, dtype=torch.long)
            self.rb_indices_per_agent.append(rb_idx)

            # DOF 全局索引（ENV 域）
            dof_list = []
            for j in range(self.dofs_per_agent):
                g = self.gym.get_actor_dof_index(env_handle, actor_i, j, gymapi.DOMAIN_ENV)
                dof_list.append(g)
            dof_idx = torch.as_tensor(dof_list, device=self.device, dtype=torch.long)
            self.dof_indices_per_agent.append(dof_idx)

            # 压缩成 slice（若连续），否则保留索引张量（后续 pack/unpack 兼容两种情况）
            s_body = _compress_to_slice(rb_idx)
            s_dof  = _compress_to_slice(dof_idx)
            self.body_slices.append(s_body if s_body is not None else rb_idx)
            self.dof_slices.append(s_dof  if s_dof  is not None else dof_idx)

        # 额外 actor 的刚体全局索引：目前我们只显式持有球的 handle
        self.extra_body_indices = {}
        if hasattr(self, "ball_handle") and self.ball_handle is not None:
            ball_rb_count = self.gym.get_actor_rigid_body_count(env_handle, self.ball_handle)
            ball_list = []
            for j in range(ball_rb_count):
                g = self.gym.get_actor_rigid_body_index(env_handle, self.ball_handle, j, gymapi.DOMAIN_ENV)
                ball_list.append(g)
            self.extra_body_indices["ball"] = torch.as_tensor(ball_list, device=self.device, dtype=torch.long)

        # 汇总（可选）：把所有机器人的刚体/DOF 索引拼成一个大表，便于一次 gather
        self.all_robot_rb_indices  = torch.cat(self.rb_indices_per_agent, dim=0) if len(self.rb_indices_per_agent) > 0 else torch.empty(0, dtype=torch.long, device=self.device)
        self.all_robot_dof_indices = torch.cat(self.dof_indices_per_agent, dim=0) if len(self.dof_indices_per_agent) > 0 else torch.empty(0, dtype=torch.long, device=self.device)

    def reset(self, 
        default_dof_pos, dof_pos, dof_vel, dof_state, 
        root_states_robot, root_states,
        last_dof_targets, last_root_vel,
        episode_length_buf,filtered_lin_vel,filtered_ang_vel,
        cmd_resample_time, delay_steps, time_out_buf,
        extras, commands, gait_frequency, dt,
        projected_gravity,base_ang_vel, gait_process, actions):

        """Reset all robots"""
        self._reset_idx(torch.arange(1, device=self.device), 
            default_dof_pos, dof_pos, dof_vel, dof_state, 
            root_states_robot, root_states,
            last_dof_targets,last_root_vel,
            episode_length_buf,filtered_lin_vel,filtered_ang_vel,
            cmd_resample_time, delay_steps,time_out_buf,
            extras)
        
        self._compute_observations(projected_gravity,base_ang_vel, commands,
            gait_frequency, gait_process,default_dof_pos,dof_pos, dof_vel, actions)
        return self.obs_buf, extras

    def _reset_idx(self, env_ids, 
        default_dof_pos, dof_pos, dof_vel,dof_state, 
        root_states_robot, root_states,
        last_dof_targets, last_root_vel,
        episode_length_buf,filtered_lin_vel,filtered_ang_vel,
        cmd_resample_time, delay_steps,time_out_buf,
        extras):

        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids, default_dof_pos, dof_pos, dof_vel,dof_state)
        self._reset_root_states(env_ids, root_states_robot, root_states)
        last_dof_targets[env_ids] = dof_pos[env_ids]
        last_root_vel[env_ids] = root_states_robot[env_ids, 7:13]
        episode_length_buf[env_ids] = 0
        filtered_lin_vel[env_ids] = 0.0
        filtered_ang_vel[env_ids] = 0.0
        cmd_resample_time[env_ids] = 0

        delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        extras["time_outs"] = time_out_buf

    def _reset_dofs(self, env_ids, 
        default_dof_pos, dof_pos, dof_vel,dof_state):

        # 1) 先把 per-agent pos/vel 填好（缓存里也更新为初值）
        for i in range(self.num_agents):
            dof_pos[i, :] = default_dof_pos[i, :]
            dof_vel[i, :] = 0.0

        # 2) 装配进全局 dof_state
        # dof_state: [..., 0]=pos, [..., 1]=vel
        for i, sl in enumerate(self.dof_slices):
            if isinstance(sl, slice):
                dof_state[sl, 0] = dof_pos[i, :]
                dof_state[sl, 1] = dof_vel[i, :]
            else:
                # 非连续索引
                dof_state.index_copy_(0, sl, torch.stack((dof_pos[i, :], dof_vel[i, :]), dim=-1))

        # 3) 一把刷回（env_ids 仍按你的外层传入，通常 [0]）
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32)),
            len(env_ids),
        )

    def _reset_root_states(self, env_ids, 
        root_states_robot, root_states):

        """
        将每个 agent 的 root state（[x,y,z,qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]）写入，
        然后一把刷回引擎。
        期望：
        - root_states_robot 形状为 [num_agents, 13] （指向全局 root_states 的前 N 行或等价视图）
        - root_states 形状为 [num_actors_total, 13]
        """
        spacing = float(self.cfg.get("env", {}).get("spawn", {}).get("spacing", 1.5))
        base_z  = float(self.base_init_state[2].item())

        for i in range(self.num_agents):
            # 平移：避免重叠；z 用模板高度，防穿地/挤飞
            root_states_robot[i, 0] = 0.0
            root_states_robot[i, 1] = i * spacing
            root_states_robot[i, 2] = base_z

            # 姿态：用模板四元数
            root_states_robot[i, 3:7] = self.base_init_state[3:7]

            # 速度清零
            root_states_robot[i, 7:13] = 0.0

        # 一把刷回
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))
    
    def _compute_observations(self,
        projected_gravity,base_ang_vel,commands,
        gait_frequency, gait_process,default_dof_pos,dof_pos, dof_vel, actions):

        """Computes observations"""
        commands_scale = torch.tensor(
            [self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["ang_vel"]],
            device=self.device,
        )
        self.obs_buf = torch.cat(
            (
                projected_gravity * self.cfg["normalization"]["gravity"],
                base_ang_vel * self.cfg["normalization"]["ang_vel"],
                commands[:, :3] * commands_scale,
                (torch.cos(2 * torch.pi * gait_process) * (gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * gait_process) * (gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (dof_pos - default_dof_pos) * self.cfg["normalization"]["dof_pos"],
                (dof_vel) * self.cfg["normalization"]["dof_vel"],
                actions,
            ),
            dim=-1,
        )
