from typing import Dict

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import get_axis_params, quat_rotate_inverse, to_torch

from envs.components.MultiAgentLowLevelController import MultiAgentLowLevelController


class BoosterT12v2Env:
    """
    Minimal 2v2 Booster T1 IsaacGym env for inference and future task development.
    Team-level semantics live here; robot physics remains in the multi-agent controller.
    """

    def __init__(self, cfg, target_xy=None):
        del target_xy
        self.controller = MultiAgentLowLevelController(cfg)
        self.cfg = cfg
        self.game_cfg = cfg.get("game", {})
        self.num_home = self.controller.num_home
        self.num_away = self.controller.num_away
        self.num_players = self.controller.num_players
        self.num_obs = cfg["env"]["num_observations"]
        self.num_privileged_obs = cfg["env"]["num_privileged_obs"]
        self.num_actions = cfg["env"]["num_actions"]
        self.num_team_obs = int(cfg["env"].get("num_team_observations", 40))
        self.dt = cfg["control"]["decimation"] * cfg["sim"]["dt"]
        self._init_buffers()

    def _init_buffers(self):
        dev = self.controller.device
        cfg = self.cfg

        actor_root_state = self.controller.gym.acquire_actor_root_state_tensor(self.controller.sim)
        dof_state_tensor = self.controller.gym.acquire_dof_state_tensor(self.controller.sim)
        body_state = self.controller.gym.acquire_rigid_body_state_tensor(self.controller.sim)
        self.controller.gym.refresh_dof_state_tensor(self.controller.sim)
        self.controller.gym.refresh_actor_root_state_tensor(self.controller.sim)
        self.controller.gym.refresh_dof_force_tensor(self.controller.sim)
        self.controller.gym.refresh_rigid_body_state_tensor(self.controller.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states_robot = self.root_states[: self.num_players, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_players, self.controller.num_dofs_per_robot, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_players, self.controller.num_dofs_per_robot, 2)[..., 1]
        body_state_tensor = gymtorch.wrap_tensor(body_state)
        total_rigid_bodies = body_state_tensor.numel() // 13
        self.body_states = body_state_tensor.view(1, total_rigid_bodies, 13)

        self.base_pos = self.root_states_robot[:, 0:3]
        self.base_quat = self.root_states_robot[:, 3:7]

        feet_indices = []
        for player_idx in range(self.num_players):
            feet_indices.extend(
                [
                    player_idx * self.controller.num_bodies_per_robot + int(local_idx.item())
                    for local_idx in self.controller.feet_indices_local
                ]
            )
        self.feet_indices = torch.tensor(feet_indices, device=dev, dtype=torch.long)
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3].view(
            1, self.num_players, len(self.controller.feet_indices_local), 3
        )

        self.obs_buf = torch.zeros(self.num_players, self.num_obs, dtype=torch.float, device=dev)
        self.team_obs_buf = torch.zeros(1, self.num_team_obs, dtype=torch.float, device=dev)
        self.rew_buf = torch.zeros(self.num_players, dtype=torch.float, device=dev)
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device=dev)
        self.time_out_buf = torch.zeros(1, dtype=torch.bool, device=dev)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.extras: Dict = {"rew_terms": {}}

        self.gravity_vec = to_torch(get_axis_params(-1.0, self.controller.up_axis_idx), device=dev).repeat((self.num_players, 1))
        self.actions = torch.zeros(self.num_players, self.num_actions, dtype=torch.float, device=dev)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_targets = torch.zeros(self.num_players, self.controller.num_dofs_per_robot, dtype=torch.float, device=dev)
        self.delay_steps = torch.zeros(self.num_players, dtype=torch.long, device=dev)
        self.torques = torch.zeros(self.num_players, self.controller.num_dofs_per_robot, dtype=torch.float, device=dev)
        self.commands = torch.zeros(self.num_players, cfg["commands"]["num_commands"], dtype=torch.float, device=dev)
        self.cmd_resample_time = torch.zeros(self.num_players, dtype=torch.long, device=dev)
        self.gait_frequency = torch.zeros(self.num_players, dtype=torch.float, device=dev)
        self.gait_process = torch.zeros(self.num_players, dtype=torch.float, device=dev)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.default_dof_pos = torch.zeros(
            self.num_players, self.controller.num_dofs_per_robot, dtype=torch.float, device=dev
        )
        for i in range(self.controller.num_dofs_per_robot):
            found = False
            for name in cfg["init_state"]["default_joint_angles"].keys():
                if name in self.controller.dof_names[i]:
                    self.default_dof_pos[:, i] = cfg["init_state"]["default_joint_angles"][name]
                    found = True
                    break
            if not found:
                self.default_dof_pos[:, i] = cfg["init_state"]["default_joint_angles"]["default"]

        self.player_names = [player["name"] for player in self.controller.player_layout]
        self.player_name_to_index = {name: idx for idx, name in enumerate(self.player_names)}
        self.ball_actor_index = self.controller.ball_actor_index
        self.ball_default_z = self.controller.ball_radius + 0.01
        self.team_indices = {
            "home": torch.arange(0, self.num_home, device=dev, dtype=torch.long),
            "away": torch.arange(self.num_home, self.num_players, device=dev, dtype=torch.long),
        }

    def reset(self):
        self.controller.reset_robots(
            self.root_states, self.root_states_robot, self.dof_pos, self.dof_vel, self.dof_state, self.default_dof_pos
        )
        self.controller.reset_ball(self.root_states)
        self.delay_steps[:] = torch.randint(
            0, self.cfg["control"]["decimation"], (self.num_players,), device=self.controller.device
        )
        self.commands.zero_()
        self.gait_frequency.zero_()
        self.gait_process.zero_()
        self.actions.zero_()
        self.last_actions.zero_()
        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        self.time_out_buf.zero_()
        self._refresh_state_tensors()
        self._compute_locomotion_obs()
        return self.obs_buf, self.extras

    def _refresh_state_tensors(self):
        self.controller.gym.refresh_actor_root_state_tensor(self.controller.sim)
        self.controller.gym.refresh_rigid_body_state_tensor(self.controller.sim)
        self.controller.gym.refresh_dof_state_tensor(self.controller.sim)
        self.base_pos[:] = self.root_states_robot[:, 0:3]
        self.base_quat[:] = self.root_states_robot[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _compute_locomotion_obs(self):
        self.obs_buf[:] = self.controller.compute_locomotion_observations(
            self.projected_gravity,
            self.base_ang_vel,
            self.commands,
            self.gait_frequency,
            self.gait_process,
            self.default_dof_pos,
            self.dof_pos,
            self.dof_vel,
            self.actions,
        )
        return self.obs_buf

    def get_ball_state(self):
        return {
            "pos": self.root_states[self.ball_actor_index, 0:3].clone(),
            "lin_vel": self.root_states[self.ball_actor_index, 7:10].clone(),
            "ang_vel": self.root_states[self.ball_actor_index, 10:13].clone(),
        }

    def compute_midlevel_obs(self):
        ball = self.get_ball_state()
        pieces = [
            self.base_pos[:, :2].reshape(-1),
            self.root_states_robot[:, 7:9].reshape(-1),
            ball["pos"][:2],
            ball["lin_vel"][:2],
        ]
        obs = torch.cat(pieces, dim=0)
        self.team_obs_buf.zero_()
        self.team_obs_buf[0, : obs.numel()] = obs
        return self.team_obs_buf

    def build_command_tensor(self, default_gait_frequency=None):
        if default_gait_frequency is None:
            default_gait_frequency = 0.5 * (
                self.cfg["commands"]["gait_frequency"][0] + self.cfg["commands"]["gait_frequency"][1]
            )
        return torch.tensor(
            [[0.0, 0.0, 0.0, float(default_gait_frequency)] for _ in range(self.num_players)],
            device=self.controller.device,
            dtype=self.commands.dtype,
        )

    def get_high_level_action_space(self):
        gait_freq = 0.5 * (
            self.cfg["commands"]["gait_frequency"][0] + self.cfg["commands"]["gait_frequency"][1]
        )
        return {name: [0.0, 0.0, 0.0, gait_freq] for name in self.player_names}

    def apply_high_level_command(self, cmd, smooth=None):
        if isinstance(cmd, dict):
            for name, values in cmd.items():
                idx = self.player_name_to_index[name]
                new_cmd = torch.tensor(values[:3], device=self.controller.device, dtype=self.commands.dtype)
                if smooth is None:
                    self.commands[idx, :3] = new_cmd
                    self.gait_frequency[idx] = float(values[3])
                else:
                    alpha = float(smooth)
                    self.commands[idx, :3] = alpha * self.commands[idx, :3] + (1.0 - alpha) * new_cmd
                    self.gait_frequency[idx] = alpha * self.gait_frequency[idx] + (1.0 - alpha) * float(values[3])
            return

        cmd_tensor = torch.as_tensor(cmd, device=self.controller.device, dtype=self.commands.dtype)
        if cmd_tensor.ndim == 1:
            cmd_tensor = cmd_tensor.view(1, -1).repeat(self.num_players, 1)
        if cmd_tensor.shape != (self.num_players, 4):
            raise ValueError(f"Expected commands of shape ({self.num_players}, 4), got {tuple(cmd_tensor.shape)}")
        if smooth is None:
            self.commands[:, :3] = cmd_tensor[:, :3]
            self.gait_frequency[:] = cmd_tensor[:, 3]
        else:
            alpha = float(smooth)
            self.commands[:, :3] = alpha * self.commands[:, :3] + (1.0 - alpha) * cmd_tensor[:, :3]
            self.gait_frequency[:] = alpha * self.gait_frequency + (1.0 - alpha) * cmd_tensor[:, 3]

    def set_team_command(self, team_name, cmd, smooth=None):
        if team_name not in self.team_indices:
            raise ValueError(f"Unknown team {team_name}")
        team_cmd = torch.as_tensor(cmd, device=self.controller.device, dtype=self.commands.dtype)
        if team_cmd.ndim == 1:
            team_cmd = team_cmd.view(1, -1).repeat(len(self.team_indices[team_name]), 1)
        if team_cmd.shape[1] != 4:
            raise ValueError(f"Expected command width 4, got {tuple(team_cmd.shape)}")
        full_cmd = self.get_high_level_action_space()
        for idx, player_idx in enumerate(self.team_indices[team_name].tolist()):
            full_cmd[self.player_names[player_idx]] = team_cmd[idx].tolist()
        self.apply_high_level_command(full_cmd, smooth=smooth)

    def pre_step(self, actions):
        self.actions[:] = torch.clip(
            actions,
            -self.cfg["normalization"]["clip_actions"],
            self.cfg["normalization"]["clip_actions"],
        )
        return self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

    def compute_reward(self):
        return torch.zeros(self.num_players, device=self.controller.device, dtype=self.base_pos.dtype)

    def _check_done(self):
        max_steps = int(self.game_cfg.get("episode_length_steps", 1000))
        return bool(self.episode_length_buf.item() >= max_steps)

    def step(self, actions):
        dof_targets = self.pre_step(actions)
        self.torques[:] = self.controller.step_low_level(
            dof_targets, self.dof_pos, self.dof_vel, self.delay_steps, self.last_dof_targets
        )
        self._refresh_state_tensors()
        self._compute_locomotion_obs()
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.episode_length_buf += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        done = self._check_done()
        self.reset_buf[:] = done
        self.time_out_buf[:] = done
        self.rew_buf[:] = self.compute_reward()
        self.extras["ball_state"] = self.get_ball_state()
        self.extras["team_obs"] = self.compute_midlevel_obs()
        self.extras["motion_metrics"] = self.summarize_motion_metrics()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_inference_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        ball = self.get_ball_state()
        players: Dict[str, Dict[str, torch.Tensor]] = {}
        for idx, name in enumerate(self.player_names):
            players[name] = {
                "base_pos": self.base_pos[idx].clone(),
                "base_quat": self.base_quat[idx].clone(),
                "base_lin_vel": self.root_states_robot[idx, 7:10].clone(),
                "base_ang_vel": self.root_states_robot[idx, 10:13].clone(),
                "command": self.commands[idx].clone(),
            }
        return {"players": players, "ball": ball}

    def summarize_motion_metrics(self) -> Dict[str, torch.Tensor]:
        world_lin_vel = self.root_states_robot[:, 7:10]
        metrics = {
            "command_xy": self.commands[:, :2].clone(),
            "command_yaw": self.commands[:, 2].clone(),
            "base_xy": self.base_pos[:, :2].clone(),
            "world_lin_vel_xy": world_lin_vel[:, :2].clone(),
            "world_yaw_rate": self.root_states_robot[:, 12].clone(),
            "speed_norm": torch.norm(world_lin_vel[:, :2], dim=-1),
            "command_speed_norm": torch.norm(self.commands[:, :2], dim=-1),
        }
        metrics["tracking_xy_error"] = metrics["world_lin_vel_xy"] - metrics["command_xy"]
        metrics["tracking_yaw_error"] = metrics["world_yaw_rate"] - metrics["command_yaw"]
        return metrics
