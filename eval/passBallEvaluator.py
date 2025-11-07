# eval/passBallEvaluator.py
from typing import Dict, Any, Optional, List
import numpy as np
import torch

from eval.evaluator import RLEvaluator, EpisodeSummary

class PassBallEvaluator(RLEvaluator):
    """
    passBall 专用评估器：
    - 以“球→目标”的距离与进步为核心指标
    - 成功依据来自 env.extras["success"]（你在 env.compute_high_level_reward() 中已设置）
    - 将 EpisodeSummary.final_dist_xy / min_dist_xy *重定义*为 “球→目标距离”（命名沿用基类）
    """

    @torch.no_grad()
    def compute_step_metrics(self, env, infos) -> Dict[str, float]:
        """从 env.extras 读取分解项；尽量容错"""
        terms = {}
        if isinstance(infos, dict):
            rt = infos.get("rew_terms", {})
            if isinstance(rt, dict):
                # 兼容你之前写入的键
                for k in (
                    "ball2goal_dist", "ball_speed_toward_goal", "ball_speed_norm",
                    "align_rb_bg", "heading_goal_term",
                    "approach_speed_rb", "speed_orth_goalline",
                    "progress_gain", "reward_total",
                    "dist_xy", "heading_cos", "spin_penalty", "time_penalty"
                ):
                    v = rt.get(k, None)
                    if v is not None:
                        try:
                            terms[k] = float(v)
                        except Exception:
                            pass

            # success 标志
            if "success" in infos:
                try:
                    terms["success_flag"] = float(1.0 if infos["success"] else 0.0)
                except Exception:
                    pass

        # 回退：若没有 ball2goal_dist，尝试用 dist_xy（不理想，但避免空值）
        if "ball2goal_dist" not in terms and "dist_xy" in terms:
            terms["ball2goal_dist"] = float(terms["dist_xy"])
        return terms

    def is_success(self, accumulator: Dict[str, Any]) -> bool:
        """优先用 success_flag；否则用球→目标距离与速度阈值近似判定"""
        if accumulator.get("success_flag", 0.0) >= 0.5:
            return True
        # 回退近似逻辑（阈值与 env 中一致/略宽松）
        d = accumulator.get("ball2goal_dist", None)
        v = accumulator.get("ball_speed_norm", None)
        if d is not None and v is not None:
            return (d < 0.35) and (v < 0.25)
        return False

    @torch.no_grad()
    def _run_one_episode(self, env, low_model, high_agent, device, tb=None) -> EpisodeSummary:
        """
        覆盖基类版本：把 accumulator 改为跟踪 **球→目标** 的距离与速度；
        最终把它们写入 EpisodeSummary 的 final_dist_xy / min_dist_xy / avg_heading_cos。
        """
        obs, infos = env.reset()
        obs = obs.to(device)
        ep_ret, steps = 0.0, 0

        acc: Dict[str, Any] = dict(
            min_ball2goal=float("inf"),
            final_ball2goal=None,
            heading_sum=0.0,
            heading_cnt=0,
            success_flag=0.0,
            ball_speed_norm=None,
        )

        while True:
            # —— 高层确定性动作（DQN=离散；SAC=连续）
            obs_high = env.compute_high_level_obs().to(device)
            obs_high_np = obs_high.squeeze(0).cpu().numpy()

            try:
                a = high_agent.select_action(obs_high_np, eval_mode=True)
            except TypeError:
                a = high_agent.select_action(obs_high_np)

            is_discrete = np.isscalar(a) or isinstance(a, (int, np.integer))
            if is_discrete:
                action_id = int(a)
                cmd = env.high_level_action_id_to_vector(action_id)
                env.apply_high_level_command(cmd)
                if tb: tb.add_scalar(f"{self.tb_prefix}/action_id", action_id)
            else:
                a_np = np.asarray(a).reshape(-1)
                a_np = np.clip(a_np, -1e3, 1e3)
                gait_freq = 1.5
                cmd = [float(a_np[0]), float(a_np[1]), float(a_np[2]), gait_freq]
                env.apply_high_level_command(cmd, smooth=0.5)
                if tb:
                    tb.add_scalar(f"{self.tb_prefix}/action_vx", float(cmd[0]))
                    tb.add_scalar(f"{self.tb_prefix}/action_vy", float(cmd[1]))
                    tb.add_scalar(f"{self.tb_prefix}/action_yaw", float(cmd[2]))

            # —— 低层（分布均值）
            obs_mod = obs.clone()
            obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
            dist = low_model.act(obs_mod)
            act = dist.loc
            obs, rew, _, infos = env.step(act)
            obs = obs.to(device)

            # —— 指标
            m = self.compute_step_metrics(env, infos)

            # reward：优先用 reward_total，回退为 env.step 的 rew
            r = float(m["reward_total"]) if "reward_total" in m else float(rew)
            ep_ret += r
            steps += 1

            # —— 跟踪“球→目标”的距离与速度
            if "ball2goal_dist" in m:
                d_bg = float(m["ball2goal_dist"])
                acc["final_ball2goal"] = d_bg
                acc["min_ball2goal"] = min(acc["min_ball2goal"], d_bg)
                if tb: tb.add_scalar(f"{self.tb_prefix}/ball2goal_dist", d_bg)

            if "ball_speed_norm" in m:
                acc["ball_speed_norm"] = float(m["ball_speed_norm"])
                if tb: tb.add_scalar(f"{self.tb_prefix}/ball_speed_norm", acc["ball_speed_norm"])

            # 朝向统计（可视化对齐度）
            if "heading_cos" in m:
                acc["heading_sum"] += float(m["heading_cos"])
                acc["heading_cnt"] += 1
                if tb: tb.add_scalar(f"{self.tb_prefix}/heading_cos", float(m["heading_cos"]))

            # success 标志
            if "success_flag" in m:
                acc["success_flag"] = max(acc["success_flag"], float(m["success_flag"]))
                if tb: tb.add_scalar(f"{self.tb_prefix}/success_flag", float(m["success_flag"]))

            if tb:
                tb.add_scalar(f"{self.tb_prefix}/reward", r)

            # —— 结束条件
            reach_max = (steps >= self.max_steps)
            success_now = self.is_success(acc)
            if reach_max or success_now:
                avg_heading = (acc["heading_sum"]/acc["heading_cnt"]) if acc["heading_cnt"]>0 else None
                # 注意：这里把 final/min 写回 EpisodeSummary 的 final_dist_xy / min_dist_xy 字段（含义是“球→目标距离”）
                return EpisodeSummary(
                    ep_return=ep_ret,
                    length=steps,
                    success=bool(success_now),
                    final_dist_xy=acc["final_ball2goal"],
                    min_dist_xy=acc["min_ball2goal"],
                    avg_heading_cos=avg_heading
                )
