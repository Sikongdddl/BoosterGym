from eval.evaluator import RLEvaluator, EpisodeSummary
from typing import Dict, Any
import numpy as np
import torch

class ChaseBallEvaluator(RLEvaluator):
    def __init__(self, max_steps: int = 200, success_dist_thresh: float = 0.4, tb_prefix: str = "eval"):
        super().__init__(max_steps=max_steps, tb_prefix=tb_prefix)
        self.success_dist_thresh = float(success_dist_thresh)

    def compute_step_metrics(self, env, infos) -> Dict[str, float]:
        # —— 基于你当前任务的定义：XY 距离 + 朝向余弦 + 指数距离奖励
        # 优先从 infos["rew_terms"] 读；缺失就自己从状态算
        m = {}
        terms = infos.get("rew_terms", {}) if isinstance(infos, dict) else {}

        # dist_xy
        if "dist_xy" in terms:
            dist_xy = float(terms["dist_xy"])
        else:
            rp = env.base_pos[0, :2].detach().cpu().numpy()
            bp = env.body_states[0, env.controller.num_bodies_robot, 0:2].detach().cpu().numpy()
            dist_xy = float(np.linalg.norm(bp - rp))
        m["dist_xy"] = dist_xy

        # heading_cos（可选）
        if "heading_cos" in terms:
            m["heading_cos"] = float(terms["heading_cos"])

        # reward（示例：exp(-dist_xy) + 0.3*heading - 0.01）
        heading_term = 0.5*(m["heading_cos"]+1.0) if "heading_cos" in m else 0.0
        m["reward"] = float(np.exp(-dist_xy) + 0.3*heading_term - 0.01)
        return m

    def is_success(self, acc: Dict[str, Any]) -> bool:
        d = acc.get("final_dist_xy", None)
        return (d is not None) and (d <= self.success_dist_thresh)
