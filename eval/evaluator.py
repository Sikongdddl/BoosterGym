# utils/evaluator.py（骨架示意）

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np, torch

@dataclass
class EpisodeSummary:
    ep_return: float
    length: int
    success: bool
    # 以下字段子类若提供就聚合；否则 None
    final_dist_xy: Optional[float] = None
    min_dist_xy: Optional[float] = None
    avg_heading_cos: Optional[float] = None

class RLEvaluator:
    def __init__(self, max_steps: int = 200, tb_prefix: str = "eval"):
        self.max_steps = int(max_steps)
        self.tb_prefix = tb_prefix

    # —— 子类必须实现：计算本任务的 step 级指标（含 reward）
    def compute_step_metrics(self, env, infos) -> Dict[str, float]:
        raise NotImplementedError

    # —— 子类必须实现：根据累积器判断成功（最后一步或累计）
    def is_success(self, accumulator: Dict[str, Any]) -> bool:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, env, low_model, high_agent, device, episodes: int = 10, tb=None):
        summaries: List[EpisodeSummary] = []
        # 如果 agent 支持 eval 模式，关掉探索
        if hasattr(high_agent, "set_eval_mode"):
            try: high_agent.set_eval_mode(True)
            except: pass

        for _ in range(episodes):
            summaries.append(self._run_one_episode(env, low_model, high_agent, device, tb))

        # —— 聚合（对 None 字段自动跳过）
        def _m(vals): 
            vs = [v for v in vals if v is not None]; 
            return float(np.mean(vs)) if vs else None

        succ = float(np.mean([int(s.success) for s in summaries])) if summaries else 0.0
        avg_ret = _m([s.ep_return for s in summaries]) or 0.0
        avg_len = _m([s.length for s in summaries]) or 0.0
        avg_fd  = _m([s.final_dist_xy for s in summaries])
        avg_md  = _m([s.min_dist_xy   for s in summaries])
        avg_hd  = _m([s.avg_heading_cos for s in summaries])

        if tb:
            p = self.tb_prefix
            tb.add_scalar(f"{p}/success_rate", succ)
            tb.add_scalar(f"{p}/avg_return",   avg_ret)
            tb.add_scalar(f"{p}/avg_length",   avg_len)
            if avg_fd is not None: tb.add_scalar(f"{p}/avg_final_dist_xy", avg_fd)
            if avg_md is not None: tb.add_scalar(f"{p}/avg_min_dist_xy",   avg_md)
            if avg_hd is not None: tb.add_scalar(f"{p}/avg_heading_cos",   avg_hd)

        return dict(
            success_rate=succ, avg_return=avg_ret, avg_length=avg_len,
            avg_final_dist_xy=avg_fd, avg_min_dist_xy=avg_md, avg_heading_cos=avg_hd,
            episodes=[s.__dict__ for s in summaries],
        )

    @torch.no_grad()
    def _run_one_episode(self, env, low_model, high_agent, device, tb=None) -> EpisodeSummary:
        obs, infos = env.reset()
        obs = obs.to(device)
        ep_ret, steps = 0.0, 0
        acc = {"min_dist_xy": float("inf"), "heading_sum": 0.0, "heading_cnt": 0, "final_dist_xy": None}

        while True:
            # —— 高层动作（贪心/确定性）
            obs_high = env.compute_high_level_obs().to(device)
            obs_high_np = obs_high.squeeze(0).cpu().numpy()
            try: action_id = high_agent.select_action(obs_high_np, eval_mode=True)
            except TypeError: action_id = high_agent.select_action(obs_high_np)
            cmd = env.high_level_action_id_to_vector(action_id)

            # —— 低层推理（均值）
            obs_mod = obs.clone()
            obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
            dist = low_model.act(obs_mod)
            act = dist.loc
            obs, _, _, infos = env.step(act)
            obs = obs.to(device)

            # —— 子类负责：计算 reward/指标
            m = self.compute_step_metrics(env, infos)
            r = float(m.get("reward", 0.0))
            ep_ret += r
            steps += 1

            # —— 累计常见诊断指标（若子类返回了就统计）
            if "dist_xy" in m:
                d = float(m["dist_xy"])
                acc["final_dist_xy"] = d
                acc["min_dist_xy"] = min(acc["min_dist_xy"], d)
                if tb: tb.add_scalar(f"{self.tb_prefix}/dist_xy", d)
            if "heading_cos" in m:
                acc["heading_sum"] += float(m["heading_cos"])
                acc["heading_cnt"] += 1
                if tb: tb.add_scalar(f"{self.tb_prefix}/heading_cos", float(m["heading_cos"]))
            if tb:
                tb.add_scalar(f"{self.tb_prefix}/reward", r)
                tb.add_scalar(f"{self.tb_prefix}/action_id", action_id)

            # —— 结束条件：步上限 or 子类判成功
            reach_max = (steps >= self.max_steps)
            success_now = self.is_success(acc)
            if reach_max or success_now:
                avg_heading = (acc["heading_sum"]/acc["heading_cnt"]) if acc["heading_cnt"]>0 else None
                return EpisodeSummary(
                    ep_return=ep_ret, length=steps, success=bool(success_now),
                    final_dist_xy=acc["final_dist_xy"], min_dist_xy=acc["min_dist_xy"], avg_heading_cos=avg_heading
                )
