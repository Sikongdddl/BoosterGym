from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from core.imitation.datasets.hypergym_vlm_dataset import ID_TO_SKILL, build_team_conditioned_observation
from envs.hyperGym.main import build_match_controller
from scripts.hlp.reward import TeamRewardShaper


@dataclass
class HyperGymSelfPlayConfig:
    num_home: int = 2
    num_away: int = 2
    max_steps: int = 120
    ball_speed: float = 0.55
    dribble_speed: float = 0.12
    wall_restitution: float = 0.0
    end_on_ball_out: bool = True
    base_seed: int = 0


def _absolute_target(state: Dict[str, Any], target_norm: np.ndarray) -> np.ndarray:
    field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
    target = np.asarray(target_norm, dtype=np.float32)
    return np.asarray([target[0] * field[0], target[1] * field[1]], dtype=np.float32)


def decode_team_action(state: Dict[str, Any], team: str, skill_ids: np.ndarray, target_norm: np.ndarray) -> Dict[str, Dict]:
    player_ids = [player["player_id"] for player in state["players"] if player["team"] == team]
    actions: Dict[str, Dict] = {}
    for idx, player_id in enumerate(player_ids):
        skill_name = ID_TO_SKILL[int(skill_ids[idx])]
        target = _absolute_target(state, target_norm[idx])
        actions[player_id] = {
            "skill": skill_name,
            "target": target,
        }
    return actions


class HyperGymPolicyOpponent:
    def __init__(self, policy, team: str, device: torch.device, deterministic: bool = True):
        self.policy = policy
        self.team = team
        self.device = device
        self.deterministic = deterministic

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Dict]:
        obs = build_team_conditioned_observation(state, controlled_team=self.team)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.act(obs_t, deterministic=self.deterministic)
        return decode_team_action(
            state,
            team=self.team,
            skill_ids=action.skill.squeeze(0).cpu().numpy(),
            target_norm=action.target.squeeze(0).cpu().numpy(),
        )


class HyperGymTeamEnv:
    def __init__(self, config: HyperGymSelfPlayConfig, reward_shaper: TeamRewardShaper | None = None):
        self.config = config
        self.reward_shaper = reward_shaper or TeamRewardShaper()
        self.episode_idx = 0
        self.controlled_team = "home"
        self.opponent_policy = None
        self.controller = None
        self.simulation = None
        self.last_state: Dict[str, Any] | None = None

    def _build_controller(self, episode_seed: int):
        self.controller = build_match_controller(
            num_home=self.config.num_home,
            num_away=self.config.num_away,
            seed=episode_seed,
            max_steps=self.config.max_steps,
            wall_restitution=self.config.wall_restitution,
            ball_speed=self.config.ball_speed,
            dribble_speed=self.config.dribble_speed,
            end_on_ball_out=self.config.end_on_ball_out,
        )
        self.simulation = self.controller.simulation

    def reset(self, *, controlled_team: str, opponent_policy, episode_seed: int | None = None) -> np.ndarray:
        self.controlled_team = controlled_team
        self.opponent_policy = opponent_policy
        if episode_seed is None:
            episode_seed = self.config.base_seed + self.episode_idx
        self._build_controller(int(episode_seed))
        self.episode_idx += 1
        self.simulation.reset()
        state = self.simulation.get_state()
        self.last_state = state
        self.reward_shaper.reset(state, team=controlled_team)
        return build_team_conditioned_observation(state, controlled_team=controlled_team)

    def step(self, skill_ids: np.ndarray, target_norm: np.ndarray):
        if self.last_state is None:
            raise RuntimeError("Call reset() before step().")
        prev_state = self.last_state
        team_action = decode_team_action(prev_state, self.controlled_team, skill_ids, target_norm)
        opponent_team = "away" if self.controlled_team == "home" else "home"
        opponent_action = self.opponent_policy(prev_state)

        if self.controlled_team == "home":
            _, env_reward, done, info = self.simulation.step(team_action, opponent_action=opponent_action)
        else:
            _, env_reward, done, info = self.simulation.step(opponent_action, opponent_action=team_action)

        next_state = info["state"]
        self.last_state = next_state
        reward = self.reward_shaper.compute_reward(
            team=self.controlled_team,
            prev_state=prev_state,
            next_state=next_state,
            events=info.get("events", []),
        )
        obs = build_team_conditioned_observation(next_state, controlled_team=self.controlled_team)
        info["team_reward"] = reward
        info["env_reward"] = env_reward
        info["controlled_team"] = self.controlled_team
        info["team_action"] = team_action
        info["opponent_action"] = opponent_action
        info["winner"] = next_state.get("winner")
        return obs, reward, done, info
