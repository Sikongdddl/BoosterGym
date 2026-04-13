import glob
import os
from typing import Dict, Optional

import numpy as np
import torch


class MidLevelPolicyManager:
    """
    Load pretrained midlevel policies and map policy_id + target -> vx/vy/theta.
    """

    def __init__(self, cfg: Dict, device: str):
        self.cfg = cfg
        self.device = device
        self.policy_cfg = cfg.get("midlevel", {})
        self.eval_mode = bool(self.policy_cfg.get("eval_mode", True))
        self._agents: Dict[str, object] = {}
        self._status: Dict[str, str] = {}
        self._load_all()

    def _resolve_checkpoint(self, checkpoint_spec: Optional[str], glob_pattern: Optional[str]) -> Optional[str]:
        if checkpoint_spec in (None, "", "null"):
            checkpoint_spec = None
        if checkpoint_spec == "-1":
            checkpoint_spec = None
        candidates = []
        if checkpoint_spec is not None:
            candidates = [checkpoint_spec] if os.path.exists(checkpoint_spec) else sorted(glob.glob(checkpoint_spec))
        elif glob_pattern:
            candidates = sorted(glob.glob(glob_pattern))
        if not candidates:
            return None
        return candidates[-1]

    def _load_agent(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if not (isinstance(ckpt, dict) and "agent" in ckpt):
            raise ValueError(f"{checkpoint_path} is not a valid SAC checkpoint with key 'agent'")
        agent = ckpt["agent"]
        agent.device = self.device
        for name in ["policy", "q1", "q2", "q1_target", "q2_target"]:
            net = getattr(agent, name, None)
            if net is not None:
                net.to(self.device)
                net.eval()
        return agent

    def _load_one(self, policy_id: str, item_cfg: Dict):
        checkpoint_path = self._resolve_checkpoint(item_cfg.get("checkpoint"), item_cfg.get("glob"))
        if checkpoint_path is None:
            self._status[policy_id] = "missing_checkpoint"
            return
        self._agents[policy_id] = self._load_agent(checkpoint_path)
        self._status[policy_id] = checkpoint_path

    def _load_all(self):
        policies_cfg = self.policy_cfg.get("policies", {})
        for policy_id, item_cfg in policies_cfg.items():
            self._load_one(policy_id, item_cfg)

    def status(self) -> Dict[str, str]:
        return dict(self._status)

    def has_policy(self, policy_id: str) -> bool:
        return policy_id in self._agents

    def act(self, policy_id: str, obs_vec, eval_mode: Optional[bool] = None) -> np.ndarray:
        if policy_id not in self._agents:
            raise KeyError(f"Midlevel policy {policy_id} is not loaded")
        agent = self._agents[policy_id]
        use_eval = self.eval_mode if eval_mode is None else bool(eval_mode)
        obs_np = np.asarray(obs_vec, dtype=np.float32)
        return agent.select_action(obs_np, eval_mode=use_eval)
