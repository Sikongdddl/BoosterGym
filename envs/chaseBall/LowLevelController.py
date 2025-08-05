import os

from isaacgym import gymtorch, gymapi
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

import numpy as np
from envs.base_task import BaseTask

from utils.utils import apply_randomization
from utils.scene import *

class LowLevelController(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.addtional_rigid_num = 0
        self._create_envs()
        self.gym.prepare_sim(self.sim)

    def _create_envs(self):
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
        ball_asset_options.density = 80
        ball_asset_options.disable_gravity = False
        ball_asset_options.fix_base_link = False
        ball_asset_options.linear_damping = 0.015
        ball_asset_options.angular_damping = 0.01
        ball_asset_options.max_angular_velocity = 100.0

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
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        # === 4. body names & index
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, booster_asset_cfg["base_name"])

        # prepare termination contact indices
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        # === 5. shape indices
        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(booster_asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(booster_asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, booster_asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        # === 6. base init state
        # only one env is needed in current job
        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        booster_start_pose = gymapi.Transform()

        # === 7. envs & actors
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(1, 4, dtype=torch.float, device=self.device)
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(1)))
        booster_start_pose.p = gymapi.Vec3(0.0,0.0,0.0) # origin
        actor_handle = self.gym.create_actor(env_handle, robot_asset, booster_start_pose, booster_asset_cfg["name"], 0, booster_asset_cfg["self_collisions"], 0)
        
        # 8 add reachable ball
        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(1,0.0,ball_radius + 0.01)
        ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "SoccerBall",0,0)
        self.addtional_rigid_num += 1

        # 9 add other assets
        self.addtional_rigid_num += create_strip_grass(self,env_handle,length=40.0,width=25.0,num_strips=15)
        self.addtional_rigid_num += create_field_boundary_lines(self,env_handle,length=40.0,width=25.0,line_width=0.15)
        self.addtional_rigid_num += create_field_auxiliary_lines(self, env_handle, length=40,width=25)

        body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
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
            body_props[j].invMass = 1.0 / body_props[j].mass

        self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
        # cancel random foot props initialization
        shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
        for idx in self.foot_shape_indices:
            shape_props[idx].friction = 1.05
            shape_props[idx].compliance = 1.0
            shape_props[idx].restitution = 0.5
        self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
        
        self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
        self.envs.append(env_handle)
        self.actor_handles.append(actor_handle)

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
        self._resample_commands(episode_length_buf, 
            cmd_resample_time, commands,gait_frequency, dt)
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

        dof_pos[env_ids] = apply_randomization(default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids, 
        root_states_robot, root_states):

        root_states_robot[env_ids] = self.base_init_state
        root_states_robot[env_ids, :2] = apply_randomization(root_states_robot[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        root_states_robot[env_ids, 2] += self.terrain.terrain_heights(root_states_robot[env_ids, :2])
        root_states_robot[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        root_states_robot[env_ids, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))

    def _resample_commands(self,
        episode_length_buf, cmd_resample_time, commands,gait_frequency,dt):

        env_ids = (episode_length_buf == cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        commands[env_ids, 0] = torch_rand_float(
            self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        commands[env_ids, 1] = torch_rand_float(
            self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        commands[env_ids, 2] = torch_rand_float(
            self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        commands[still_envs, :] = 0.0
        gait_frequency[still_envs] = 0.0
        cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / dt),
            (len(env_ids),),
            device=self.device,
        )

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
