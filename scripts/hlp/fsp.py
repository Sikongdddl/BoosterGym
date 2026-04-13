from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from envs.hyperGym.policies import SimpleMatchPolicy
from scripts.hlp.env import HyperGymPolicyOpponent
from scripts.hlp.policy import TeamHybridActorCritic


@dataclass
class PolicySnapshot:
    label: str
    state_dict: dict


class FSPOpponentPool:
    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: torch.device,
        scripted_prob: float = 0.2,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.scripted_prob = float(np.clip(scripted_prob, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.snapshots: List[PolicySnapshot] = []

    def add_snapshot(self, policy: TeamHybridActorCritic, label: str) -> None:
        frozen = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        self.snapshots.append(PolicySnapshot(label=label, state_dict=frozen))

    def sample_opponent(self, team: str, deterministic: bool = True):
        if not self.snapshots or self.rng.random() < self.scripted_prob:
            return SimpleMatchPolicy(team=team, seed=int(self.rng.integers(0, 1_000_000)))

        snapshot = self.snapshots[int(self.rng.integers(0, len(self.snapshots)))]
        policy = TeamHybridActorCritic(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        policy.load_state_dict(copy.deepcopy(snapshot.state_dict))
        policy.eval()
        return HyperGymPolicyOpponent(policy=policy, team=team, device=self.device, deterministic=deterministic)
