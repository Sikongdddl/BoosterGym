# utils/evaluator.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os, time, numpy as np, torch

@dataclass
class EpisodeSummary:
    ep_return: float
    length: int
    success: bool
    final_dist_xy: Optional[float] = None
    min_dist_xy: Optional[float] = None
    avg_heading_cos: Optional[float] = None

class RLEvaluator:
    def __init__(
        self,
        max_steps: int = 200,
        tb_prefix: str = "eval",
        # —— 仅与保存权重相关的可选项 —— #
        save_dir: Optional[str] = None,            # 目录，不填则不保存
        save_best_by: Optional[str] = None,        # 根据该指标保存 best；None 表示不保存 best
        higher_is_better: bool = True,             # 该指标是越大越好？
        save_every_eval: bool = False,             # 每次评估都保存 last_*.pth
    ):
        self.max_steps = int(max_steps)
        self.tb_prefix = tb_prefix

        self.save_dir = save_dir
        self.save_best_by = save_best_by
        self.higher_is_better = higher_is_better
        self.save_every_eval = save_every_eval
        self._best_metric: Optional[float] = None

    # —— 子类实现 —— #
    def compute_step_metrics(self, env, infos) -> Dict[str, float]:
        raise NotImplementedError

    def is_success(self, accumulator: Dict[str, Any]) -> bool:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, env, low_model, high_agent, device, episodes: int = 10, tb=None,
                 global_step: Optional[int] = None) -> Dict[str, Any]:
        summaries: List[EpisodeSummary] = []

        # 可选：进入“评估模式”
        if hasattr(high_agent, "set_eval_mode"):
            try: high_agent.set_eval_mode(True)
            except: pass

        for _ in range(episodes):
            summaries.append(self._run_one_episode(env, low_model, high_agent, device, tb))

        # 聚合
        def _m(vals):
            vs = [v for v in vals if v is not None]
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

        results = dict(
            success_rate=succ, avg_return=avg_ret, avg_length=avg_len,
            avg_final_dist_xy=avg_fd, avg_min_dist_xy=avg_md, avg_heading_cos=avg_hd,
            episodes=[s.__dict__ for s in summaries],
        )

        # —— 只保存“权重” —— #
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            step_str = f"step{global_step}" if global_step is not None else "nostep"
            weights = self._collect_weights(low_model, high_agent)

            if self.save_every_eval:
                torch.save(weights, os.path.join(self.save_dir, f"last_{step_str}.pth"))

            if self.save_best_by is not None:
                metric = results.get(self.save_best_by, None)
                if metric is not None:
                    is_better = (self._best_metric is None) or \
                                (metric > self._best_metric if self.higher_is_better else metric < self._best_metric)
                    if is_better:
                        self._best_metric = metric
                        torch.save(weights, os.path.join(self.save_dir, f"best_{self.save_best_by}_{step_str}.pth"))

        return results

    @torch.no_grad()
    def _run_one_episode(self, env, low_model, high_agent, device, tb=None) -> EpisodeSummary:
        obs, infos = env.reset()
        obs = obs.to(device)
        ep_ret, steps = 0.0, 0
        acc = {"min_dist_xy": float("inf"), "heading_sum": 0.0, "heading_cnt": 0, "final_dist_xy": None}

        while True:
            # 高层确定性动作
            obs_high = env.compute_high_level_obs().to(device)
            obs_high_np = obs_high.squeeze(0).cpu().numpy()
            try:
                action_id = high_agent.select_action(obs_high_np, eval_mode=True)
            except TypeError:
                action_id = high_agent.select_action(obs_high_np)
            cmd = env.high_level_action_id_to_vector(action_id)

            # 低层（分布均值）
            obs_mod = obs.clone()
            obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
            dist = low_model.act(obs_mod)
            act = dist.loc
            obs, rew, _, infos = env.step(act)
            obs = obs.to(device)

            # 子类指标（不二次调用 reward）
            m = self.compute_step_metrics(env, infos)
            r = float(rew) if "reward" not in m else float(m["reward"])
            ep_ret += r
            steps += 1

            # 诊断累计
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

            # 结束
            reach_max = (steps >= self.max_steps)
            success_now = self.is_success(acc)
            if reach_max or success_now:
                avg_heading = (acc["heading_sum"]/acc["heading_cnt"]) if acc["heading_cnt"]>0 else None
                return EpisodeSummary(
                    ep_return=ep_ret, length=steps, success=bool(success_now),
                    final_dist_xy=acc["final_dist_xy"], min_dist_xy=acc["min_dist_xy"], avg_heading_cos=avg_heading
                )

    # ---------- 只收集“权重”的通用方法 ----------
    def _collect_weights(self, low_model, high_agent) -> Dict[str, Any]:
        """仅收集具有 state_dict() 的对象的权重。兼容任意算法/任务。"""
        weights: Dict[str, Any] = {}

        def try_take(name: str, obj: Any):
            if obj is None: return
            sd = None
            try:
                if hasattr(obj, "state_dict"):
                    sd = obj.state_dict()
            except Exception:
                sd = None
            if sd is not None:
                weights[name] = sd

        try_take("low_model", low_model)
        try_take("high_agent", high_agent)  # 若 agent 本身是 nn.Module
        # 常见容器字段（可选；不存在就跳过）
        for name in ("policy", "q_net", "actor", "critic", "value", "target_net"):
            if hasattr(high_agent, name):
                try_take(f"high_agent.{name}", getattr(high_agent, name))

        return weights
