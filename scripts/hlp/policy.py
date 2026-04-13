from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.5


@dataclass
class PolicyAction:
    skill: torch.Tensor
    target: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class TeamHybridActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_players: int = 2,
        num_skills: int = 3,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.skill_head = nn.Linear(in_dim, num_players * num_skills)
        self.target_mean_head = nn.Linear(in_dim, num_players * 2)
        self.target_log_std_head = nn.Linear(in_dim, num_players * 2)
        self.value_head = nn.Linear(in_dim, 1)
        self.num_players = num_players
        self.num_skills = num_skills

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(obs)
        return {
            "hidden": hidden,
            "skill_logits": self.skill_head(hidden).view(-1, self.num_players, self.num_skills),
            "target_mean": self.target_mean_head(hidden).view(-1, self.num_players, 2),
            "target_log_std": self.target_log_std_head(hidden).view(-1, self.num_players, 2).clamp(LOG_STD_MIN, LOG_STD_MAX),
            "value": self.value_head(hidden).squeeze(-1),
        }

    def _sample_target(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        squashed = torch.tanh(pre_tanh)
        target = 0.5 * (squashed + 1.0)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=(-1, -2))
        entropy = dist.entropy().sum(dim=(-1, -2))
        return target, log_prob, entropy

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> PolicyAction:
        out = self.forward(obs)
        skill_dist = Categorical(logits=out["skill_logits"])
        if deterministic:
            skill = out["skill_logits"].argmax(dim=-1)
        else:
            skill = skill_dist.sample()
        skill_log_prob = skill_dist.log_prob(skill).sum(dim=-1)
        skill_entropy = skill_dist.entropy().sum(dim=-1)
        target, target_log_prob, target_entropy = self._sample_target(
            out["target_mean"],
            out["target_log_std"],
            deterministic=deterministic,
        )
        return PolicyAction(
            skill=skill,
            target=target,
            log_prob=skill_log_prob + target_log_prob,
            value=out["value"],
            entropy=skill_entropy + target_entropy,
        )

    def evaluate_actions(self, obs: torch.Tensor, skill: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(obs)
        skill_dist = Categorical(logits=out["skill_logits"])
        skill_log_prob = skill_dist.log_prob(skill).sum(dim=-1)
        skill_entropy = skill_dist.entropy().sum(dim=-1)

        target_clamped = target.clamp(1e-4, 1.0 - 1e-4)
        squashed = target_clamped * 2.0 - 1.0
        pre_tanh = 0.5 * torch.log((1.0 + squashed) / (1.0 - squashed))
        target_dist = Normal(out["target_mean"], out["target_log_std"].exp())
        target_log_prob = target_dist.log_prob(pre_tanh) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        target_log_prob = target_log_prob.sum(dim=(-1, -2))
        target_entropy = target_dist.entropy().sum(dim=(-1, -2))
        return {
            "log_prob": skill_log_prob + target_log_prob,
            "entropy": skill_entropy + target_entropy,
            "value": out["value"],
        }

    def load_bc_actor(self, checkpoint_path: str, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        remapped_state = {}
        for key, value in state_dict.items():
            if key.startswith("target_head."):
                remapped_state[key.replace("target_head.", "target_mean_head.")] = value
            else:
                remapped_state[key] = value
        own_state = self.state_dict()
        transferable = {
            key: value
            for key, value in remapped_state.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        self.load_state_dict(transferable, strict=False)
