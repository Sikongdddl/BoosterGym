import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, to_torch

from envs.base_task import BaseTask
from utils.scene import create_field_auxiliary_lines, create_field_boundary_lines, create_strip_grass


class MultiAgentLowLevelController(BaseTask):
    """
    Independent multi-robot scene/controller for 2v2 Booster T1.
    It intentionally does not mutate the legacy single-robot LowLevelController.
    """

    def __init__(self, cfg):
        self.game_cfg = cfg.get("game", {})
        self.num_home = int(self.game_cfg.get("num_home", 2))
        self.num_away = int(self.game_cfg.get("num_away", 2))
        self.num_players = self.num_home + self.num_away
        self.enable_field_decor = bool(self.game_cfg.get("enable_field_decor", False))
        super().__init__(cfg)
        self.additional_rigid_num = 0
        self._create_envs()
        self.gym.prepare_sim(self.sim)

    def _load_robot_asset(self):
        booster_asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(booster_asset_cfg["file"])
        asset_file = os.path.basename(booster_asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = booster_asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = booster_asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = booster_asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = booster_asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = booster_asset_cfg["fix_base_link"]
        asset_options.density = booster_asset_cfg["density"]
        asset_options.angular_damping = booster_asset_cfg["angular_damping"]
        asset_options.linear_damping = booster_asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = booster_asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = booster_asset_cfg["max_linear_velocity"]
        asset_options.armature = booster_asset_cfg["armature"]
        asset_options.thickness = booster_asset_cfg["thickness"]
        asset_options.disable_gravity = booster_asset_cfg["disable_gravity"]

        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _create_ball_asset(self):
        ball_radius = float(self.game_cfg.get("ball_radius", 0.11))
        ball_cfg = self.game_cfg.get("ball_dynamics", {})
        ball_options = gymapi.AssetOptions()
        ball_options.density = float(ball_cfg.get("density", 80.0))
        ball_options.disable_gravity = False
        ball_options.fix_base_link = False
        ball_options.linear_damping = float(ball_cfg.get("linear_damping", 0.015))
        ball_options.angular_damping = float(ball_cfg.get("angular_damping", 0.01))
        ball_options.max_angular_velocity = float(ball_cfg.get("max_angular_velocity", 100.0))

        self.ball_radius = ball_radius
        self.ball_density = float(ball_options.density)
        self.ball_linear_damping = float(ball_options.linear_damping)
        self.ball_angular_damping = float(ball_options.angular_damping)
        return self.gym.create_sphere(self.sim, ball_radius, ball_options)

    def _compute_spawn_layout(self):
        spawn_cfg = self.game_cfg.get("spawn", {})
        home_x = float(spawn_cfg.get("home_x", -2.0))
        away_x = float(spawn_cfg.get("away_x", 2.0))
        y_offsets = spawn_cfg.get("y_offsets", [-0.9, 0.9])
        if len(y_offsets) < max(self.num_home, self.num_away):
            raise ValueError("game.spawn.y_offsets must cover all players per team")
        base_z = float(self.cfg["init_state"]["pos"][2])

        layout = []
        for idx in range(self.num_home):
            layout.append(
                {
                    "name": f"home_{idx}",
                    "team": "home",
                    "xy": (home_x, float(y_offsets[idx])),
                    "yaw": 0.0,
                }
            )
        for idx in range(self.num_away):
            layout.append(
                {
                    "name": f"away_{idx}",
                    "team": "away",
                    "xy": (away_x, float(y_offsets[idx])),
                    "yaw": np.pi,
                }
            )
        for item in layout:
            item["z"] = base_z
        return layout

    def _create_envs(self):
        booster_asset_cfg = self.cfg["asset"]
        robot_asset = self._load_robot_asset()
        ball_asset = self._create_ball_asset()

        self.num_dofs_per_robot = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies_per_robot = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs_per_robot, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs_per_robot, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs_per_robot, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs_per_robot):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(1, self.num_dofs_per_robot, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(1, self.num_dofs_per_robot, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(1, self.num_dofs_per_robot, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs_per_robot):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
                    break
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")

        self.base_body_index = self.gym.find_asset_rigid_body_index(robot_asset, booster_asset_cfg["base_name"])
        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices_local = torch.zeros(len(booster_asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices_local = []
        for i, foot_name in enumerate(booster_asset_cfg["foot_names"]):
            body_index = self.gym.find_asset_rigid_body_index(robot_asset, foot_name)
            self.feet_indices_local[i] = body_index
            self.foot_shape_indices_local += list(
                range(rbs_list[body_index].start, rbs_list[body_index].start + rbs_list[body_index].count)
            )

        self.base_init_state = to_torch(
            self.cfg["init_state"]["pos"]
            + self.cfg["init_state"]["rot"]
            + self.cfg["init_state"]["lin_vel"]
            + self.cfg["init_state"]["ang_vel"],
            device=self.device,
        )

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.robot_actor_handles = []
        self.robot_actor_indices = []
        self.player_layout = self._compute_spawn_layout()

        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(1)))
        self.envs.append(env_handle)

        for player_idx, player in enumerate(self.player_layout):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(player["xy"][0], player["xy"][1], player["z"])
            quat = quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([player["yaw"]], device=self.device),
            )[0]
            pose.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                pose,
                player["name"],
                0,
                booster_asset_cfg["self_collisions"],
                0,
            )
            actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
            self.actor_handles.append(actor_handle)
            self.robot_actor_handles.append(actor_handle)
            self.robot_actor_indices.append(actor_index)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            for body_idx, body_prop in enumerate(body_props):
                body_prop.com.x = 0.0
                body_prop.com.y = 0.0
                body_prop.com.z = 0.0
                body_prop.mass = 1.0
                body_prop.invMass = 1.0 / body_prop.mass
                body_props[body_idx] = body_prop
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            for idx in self.foot_shape_indices_local:
                shape_props[idx].friction = 1.05
                shape_props[idx].compliance = 1.0
                shape_props[idx].restitution = 0.5
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(0.0, 0.0, self.ball_radius + 0.01)
        self.ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "SoccerBall", 0, 0)
        self.ball_actor_index = self.gym.get_actor_index(env_handle, self.ball_handle, gymapi.DOMAIN_SIM)
        self.actor_handles.append(self.ball_handle)
        self.additional_rigid_num += 1

        if self.enable_field_decor:
            self.additional_rigid_num += create_strip_grass(self, env_handle, length=40.0, width=25.0, num_strips=15)
            self.additional_rigid_num += create_field_boundary_lines(self, env_handle, length=40.0, width=25.0, line_width=0.15)
            self.additional_rigid_num += create_field_auxiliary_lines(self, env_handle, length=40.0, width=25.0)

        self.robot_actor_indices = torch.tensor(self.robot_actor_indices, device=self.device, dtype=torch.long)
        self.total_num_dofs = self.num_players * self.num_dofs_per_robot
        self.total_num_robot_bodies = self.num_players * self.num_bodies_per_robot
        self.num_actors = self.num_players + 1

    def reset_robots(self, root_states, root_states_robot, dof_pos, dof_vel, dof_state, default_dof_pos):
        for idx, player in enumerate(self.player_layout):
            root_states_robot[idx] = self.base_init_state
            root_states_robot[idx, 0] = player["xy"][0]
            root_states_robot[idx, 1] = player["xy"][1]
            root_states_robot[idx, 2] = player["z"]
            quat = quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([player["yaw"]], device=self.device),
            )[0]
            root_states_robot[idx, 3:7] = quat
            root_states_robot[idx, 7:13] = 0.0

        dof_pos[:] = default_dof_pos
        dof_vel[:] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state))

    def reset_ball(self, root_states):
        root_states[self.ball_actor_index, 0] = 0.0
        root_states[self.ball_actor_index, 1] = 0.0
        root_states[self.ball_actor_index, 2] = self.ball_radius + 0.01
        root_states[self.ball_actor_index, 3:7] = 0.0
        root_states[self.ball_actor_index, 6] = 1.0
        root_states[self.ball_actor_index, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))

    def step_low_level(self, dof_targets, dof_pos, dof_vel, delay_steps, last_dof_targets):
        torques = torch.zeros_like(dof_targets)
        for i in range(self.cfg["control"]["decimation"]):
            mask = delay_steps == i
            if torch.any(mask):
                last_dof_targets[mask] = dof_targets[mask]
            dof_torques = self.dof_stiffness * (last_dof_targets - dof_pos) - self.dof_damping * dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques.reshape(-1)))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        if getattr(self, "viewer", None) is not None:
            self.render()
        return torques / self.cfg["control"]["decimation"]

    def compute_locomotion_observations(
        self,
        projected_gravity,
        base_ang_vel,
        commands,
        gait_frequency,
        gait_process,
        default_dof_pos,
        dof_pos,
        dof_vel,
        actions,
    ):
        commands_scale = torch.tensor(
            [
                self.cfg["normalization"]["lin_vel"],
                self.cfg["normalization"]["lin_vel"],
                self.cfg["normalization"]["ang_vel"],
            ],
            device=self.device,
        )
        return torch.cat(
            (
                projected_gravity * self.cfg["normalization"]["gravity"],
                base_ang_vel * self.cfg["normalization"]["ang_vel"],
                commands[:, :3] * commands_scale,
                (torch.cos(2 * torch.pi * gait_process) * (gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * gait_process) * (gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (dof_pos - default_dof_pos) * self.cfg["normalization"]["dof_pos"],
                dof_vel * self.cfg["normalization"]["dof_vel"],
                actions,
            ),
            dim=-1,
        )
