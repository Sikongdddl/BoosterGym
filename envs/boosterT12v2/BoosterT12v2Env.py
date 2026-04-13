from typing import Dict

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import get_axis_params, quat_from_euler_xyz, quat_rotate, quat_rotate_inverse, to_torch

from envs.components.MidLevelPolicyManager import MidLevelPolicyManager
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
        self.midlevel_policy = MidLevelPolicyManager(cfg, device=self.controller.device)
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
        self.last_policy_actions = self.get_policy_action_space()

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

    def set_from_hyper_state(self, state: Dict):
        state_field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
        sim_field_cfg = self.game_cfg.get("field", {})
        sim_field = np.asarray(
            [
                float(sim_field_cfg.get("length", 14.0)),
                float(sim_field_cfg.get("width", 9.0)),
            ],
            dtype=np.float32,
        )

        def map_xy(xy):
            arr = np.asarray(xy, dtype=np.float32)
            nx = 0.0 if state_field[0] <= 1e-6 else float(arr[0]) / float(state_field[0])
            ny = 0.0 if state_field[1] <= 1e-6 else float(arr[1]) / float(state_field[1])
            wx = (nx - 0.5) * float(sim_field[0])
            wy = (ny - 0.5) * float(sim_field[1])
            return wx, wy

        def map_vel(vxy):
            arr = np.asarray(vxy, dtype=np.float32)
            sx = float(sim_field[0]) / max(float(state_field[0]), 1e-6)
            sy = float(sim_field[1]) / max(float(state_field[1]), 1e-6)
            return float(arr[0]) * sx, float(arr[1]) * sy

        self.controller.reset_robots(
            self.root_states, self.root_states_robot, self.dof_pos, self.dof_vel, self.dof_state, self.default_dof_pos
        )
        self.controller.reset_ball(self.root_states)

        players = state.get("players", [])
        if len(players) != self.num_players:
            raise ValueError(f"Expected {self.num_players} players, got {len(players)}")

        for idx, player in enumerate(players):
            wx, wy = map_xy(player.get("position", [0.0, 0.0]))
            vx, vy = map_vel(player.get("velocity", [0.0, 0.0]))
            yaw = float(player.get("heading", 0.0))
            quat = quat_from_euler_xyz(
                torch.tensor([0.0], device=self.controller.device),
                torch.tensor([0.0], device=self.controller.device),
                torch.tensor([yaw], device=self.controller.device),
            )[0]
            self.root_states_robot[idx, 0] = wx
            self.root_states_robot[idx, 1] = wy
            self.root_states_robot[idx, 2] = float(self.controller.base_init_state[2].item())
            self.root_states_robot[idx, 3:7] = quat
            self.root_states_robot[idx, 7] = vx
            self.root_states_robot[idx, 8] = vy
            self.root_states_robot[idx, 9:13] = 0.0

        ball_pos = state.get("ball_position", [0.0, 0.0])
        ball_vel = state.get("ball_velocity", [0.0, 0.0])
        bwx, bwy = map_xy(ball_pos)
        bvx, bvy = map_vel(ball_vel)
        self.root_states[self.ball_actor_index, 0] = bwx
        self.root_states[self.ball_actor_index, 1] = bwy
        self.root_states[self.ball_actor_index, 2] = self.ball_default_z
        self.root_states[self.ball_actor_index, 3:7] = 0.0
        self.root_states[self.ball_actor_index, 6] = 1.0
        self.root_states[self.ball_actor_index, 7] = bvx
        self.root_states[self.ball_actor_index, 8] = bvy
        self.root_states[self.ball_actor_index, 9:13] = 0.0

        self.controller.gym.set_actor_root_state_tensor(self.controller.sim, gymtorch.unwrap_tensor(self.root_states))
        self.controller.gym.set_dof_state_tensor(self.controller.sim, gymtorch.unwrap_tensor(self.dof_state))
        self._refresh_state_tensors()
        self._compute_locomotion_obs()

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

    def get_policy_action_space(self):
        ball_state = self.get_ball_state()
        ball_xy = ball_state["pos"][:2]
        return {
            name: {
                "policy_id": "move_to_target",
                "target": [float(ball_xy[0]), float(ball_xy[1])],
            }
            for name in self.player_names
        }

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

    def _compute_chase_obs_for_player(self, player_idx: int, target_xy: torch.Tensor) -> torch.Tensor:
        robot_pos = self.base_pos[player_idx, :3]
        delta_world = torch.zeros(3, device=self.controller.device, dtype=robot_pos.dtype)
        delta_world[:2] = target_xy - robot_pos[:2]
        delta_body = quat_rotate_inverse(self.base_quat[player_idx : player_idx + 1], delta_world[None, :]).squeeze(0)
        delta_xy_body = delta_body[:2]
        dist_xy = torch.norm(delta_xy_body) + 1e-6
        bearing = torch.atan2(delta_xy_body[1], delta_xy_body[0])
        cos_b = torch.cos(bearing)
        sin_b = torch.sin(bearing)
        v_body_xy = self.base_lin_vel[player_idx, :2]
        speed_toward = v_body_xy[0] * cos_b + v_body_xy[1] * sin_b
        return torch.stack(
            [
                delta_xy_body[0],
                delta_xy_body[1],
                dist_xy,
                cos_b,
                sin_b,
                v_body_xy[0],
                v_body_xy[1],
                speed_toward,
            ],
            dim=0,
        )

    def _compute_pass_obs_for_player(self, player_idx: int, target_xy: torch.Tensor) -> torch.Tensor:
        base_pos = self.base_pos[player_idx, :3]
        base_x = base_pos[0]
        base_y = base_pos[1]
        ball_pos = self.root_states[self.ball_actor_index, 0:3]
        ball_xy = ball_pos[:2]
        ball_x = ball_xy[0]
        ball_y = ball_xy[1]
        v_body = self.base_lin_vel[player_idx, :3]
        v_world = quat_rotate(self.base_quat[player_idx : player_idx + 1], v_body[None, :]).squeeze(0)
        delta_world = torch.zeros(3, device=self.controller.device, dtype=base_pos.dtype)
        delta_world[0] = ball_x - base_x
        delta_world[1] = ball_y - base_y
        delta_body = quat_rotate_inverse(self.base_quat[player_idx : player_idx + 1], delta_world[None, :]).squeeze(0)
        rel_ball_x = delta_body[0]
        rel_ball_y = delta_body[1]
        rel_ball_ang = torch.atan2(rel_ball_y, rel_ball_x + 1e-6)
        return torch.stack(
            [
                base_x,
                base_y,
                ball_x,
                ball_y,
                v_world[0],
                v_world[1],
                target_xy[0],
                target_xy[1],
                rel_ball_x,
                rel_ball_y,
                rel_ball_ang,
            ],
            dim=0,
        )

    def _policy_target_to_tensor(self, target) -> torch.Tensor:
        target_tensor = torch.as_tensor(target, device=self.controller.device, dtype=self.base_pos.dtype)
        if target_tensor.numel() < 2:
            raise ValueError(f"Invalid target {target}")
        return target_tensor[:2]

    def policy_action_to_command(self, player_name: str, policy_action: Dict, eval_mode=None) -> np.ndarray:
        player_idx = self.player_name_to_index[player_name]
        policy_id = str(policy_action.get("policy_id", "move_to_target"))
        target_xy = self._policy_target_to_tensor(policy_action.get("target", self.base_pos[player_idx, :2]))

        obs_builder = {
            "move_to_target": self._compute_chase_obs_for_player,
            "dribble_to_target": self._compute_chase_obs_for_player,
            "pass_to_target": self._compute_pass_obs_for_player,
        }
        routed_policy = {
            "move_to_target": "move_to_target",
            "dribble_to_target": "move_to_target",
            "pass_to_target": "pass_to_target",
        }.get(policy_id)

        if routed_policy is None or routed_policy not in obs_builder:
            return np.asarray([0.0, 0.0, 0.0, float(self.gait_frequency[player_idx].item())], dtype=np.float32)

        if not self.midlevel_policy.has_policy(routed_policy):
            return np.asarray([0.0, 0.0, 0.0, float(self.gait_frequency[player_idx].item())], dtype=np.float32)

        obs_vec = obs_builder[routed_policy](player_idx, target_xy)
        action_xyz = self.midlevel_policy.act(routed_policy, obs_vec.detach().cpu().numpy(), eval_mode=eval_mode)
        gait_freq = 0.5 * (
            self.cfg["commands"]["gait_frequency"][0] + self.cfg["commands"]["gait_frequency"][1]
        )
        return np.asarray([float(action_xyz[0]), float(action_xyz[1]), float(action_xyz[2]), gait_freq], dtype=np.float32)

    def apply_policy_command(self, policy_actions: Dict[str, Dict], smooth=None, eval_mode=None):
        cmd = {}
        for player_name in self.player_names:
            policy_action = policy_actions.get(player_name, {"policy_id": "move_to_target", "target": self.base_pos[self.player_name_to_index[player_name], :2]})
            cmd[player_name] = self.policy_action_to_command(player_name, policy_action, eval_mode=eval_mode).tolist()
        self.last_policy_actions = {
            player_name: {
                "policy_id": str(policy_actions.get(player_name, {}).get("policy_id", "move_to_target")),
                "target": list(np.asarray(policy_actions.get(player_name, {}).get("target", self.base_pos[self.player_name_to_index[player_name], :2].detach().cpu().numpy()), dtype=np.float32)[:2]),
            }
            for player_name in self.player_names
        }
        self.apply_high_level_command(cmd, smooth=smooth)

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
        self.extras["midlevel_policy_status"] = self.midlevel_policy.status()
        self.extras["policy_actions"] = self.last_policy_actions
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
