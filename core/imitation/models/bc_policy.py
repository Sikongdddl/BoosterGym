from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TeamActionBCPolicy(nn.Module):
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
        in_dim = int(obs_dim)
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.skill_head = nn.Linear(in_dim, num_players * num_skills)
        self.target_head = nn.Linear(in_dim, num_players * 2)
        self.num_players = int(num_players)
        self.num_skills = int(num_skills)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(obs)
        skill_logits = self.skill_head(hidden).view(-1, self.num_players, self.num_skills)
        target_pred = self.target_head(hidden).view(-1, self.num_players, 2)
        return {
            "skill_logits": skill_logits,
            "target_pred": target_pred,
        }
