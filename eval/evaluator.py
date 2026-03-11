from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import json
import os
import time

import numpy as np
import torch


@dataclass
class StepRecord:
    step: int
    reward: float
    robot_xy: Tuple[float, float]
    ball_xy: Tuple[float, float]
    target_xy: Optional[Tuple[float, float]]
    robot_v_xy: Tuple[float, float]
    ball_v_xy: Tuple[float, float]
    success: bool
    fail: bool
    fall: bool
    hit: bool
    rew_terms: Dict[str, float]
    task: Dict[str, float]


@dataclass
class EpisodeSummary:
    episode_index: int
    ep_return: float
    length: int
    success: bool
    fail: bool
    fall: bool
    hit: bool
    hit_step: Optional[int]
    success_step: Optional[int]
    fail_step: Optional[int]
    final_robot_xy: Tuple[float, float]
    final_ball_xy: Tuple[float, float]
    final_target_xy: Optional[Tuple[float, float]]
    initial_robot_to_ball_dist: Optional[float]
    min_robot_to_ball_dist: Optional[float]
    final_robot_to_ball_dist: Optional[float]
    task_metrics: Dict[str, float]


@dataclass
class EpisodeRecord:
    summary: EpisodeSummary
    steps: List[StepRecord]


class RLEvaluator:
    def __init__(
        self,
        max_steps: int = 300,
        action_repeat: int = 10,
        gait_frequency: float = 1.5,
        tb_prefix: str = "eval",
        save_dir: Optional[str] = None,
        render_plots: bool = True,
    ):
        self.max_steps = int(max_steps)
        self.action_repeat = int(action_repeat)
        self.gait_frequency = float(gait_frequency)
        self.tb_prefix = tb_prefix
        self.save_dir = save_dir
        self.render_plots = render_plots

    @property
    def task_name(self) -> str:
        return self.__class__.__name__.replace("Evaluator", "")

    def get_high_level_obs(self, env):
        if hasattr(env, "compute_midlevel_obs"):
            return env.compute_midlevel_obs()
        if hasattr(env, "compute_high_level_obs"):
            return env.compute_high_level_obs()
        raise AttributeError("Env must implement compute_midlevel_obs() or compute_high_level_obs().")

    def get_gait_frequency(self, env) -> float:
        if hasattr(env, "get_high_level_action_space"):
            try:
                return float(env.get_high_level_action_space()[-1])
            except Exception:
                pass
        return self.gait_frequency

    def select_high_level_action(self, high_agent, obs_high_np: np.ndarray):
        try:
            return high_agent.select_action(obs_high_np, eval_mode=True)
        except TypeError:
            return high_agent.select_action(obs_high_np)

    def action_to_command(self, env, action) -> List[float]:
        gait_freq = self.get_gait_frequency(env)
        is_discrete = np.isscalar(action) or isinstance(action, (int, np.integer))
        if is_discrete:
            action_id = int(action)
            cmd = list(env.high_level_action_id_to_vector(action_id))
            return [float(cmd[0]), float(cmd[1]), float(cmd[2]), gait_freq]

        a_np = np.asarray(action, dtype=np.float32).reshape(-1)
        if a_np.size < 3:
            raise ValueError(f"Continuous action dim must be >=3, got shape {a_np.shape}")
        return [float(a_np[0]), float(a_np[1]), float(a_np[2]), gait_freq]

    def apply_command(self, env, cmd: Sequence[float]):
        env.apply_high_level_command(list(cmd), smooth=0.5)

    def rollout_low_level(self, env, low_model, obs, cmd, device):
        last_infos: Dict[str, Any] = {}
        total_reward = 0.0
        done_flag = False

        for _ in range(self.action_repeat):
            obs_mod = obs.clone()
            obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
            dist = low_model.act(obs_mod)
            act = dist.loc
            obs, rew, done, infos = env.step(act)
            obs = obs.to(device)
            total_reward += float(rew.reshape(-1)[0].item() if torch.is_tensor(rew) else rew)
            last_infos = infos if isinstance(infos, dict) else {}
            done_flag = bool(torch.any(done).item()) if torch.is_tensor(done) else bool(done)

            if self.should_break_rollout(last_infos, done_flag):
                break

        return obs, total_reward / float(self.action_repeat), done_flag, last_infos

    def should_break_rollout(self, infos: Dict[str, Any], done_flag: bool) -> bool:
        return done_flag or any(bool(infos.get(k, False)) for k in ("success", "fail", "fall"))

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            value = value.detach().reshape(-1)[0].item()
        try:
            return float(value)
        except Exception:
            return None

    def _xy_from_tensor(self, value: Any) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy().reshape(-1)
        else:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return None
        return (float(arr[0]), float(arr[1]))

    def _extract_common_step(self, env, infos: Dict[str, Any], reward: float, step_idx: int) -> StepRecord:
        robot_xy = self._xy_from_tensor(env.base_pos[0, :2]) or (0.0, 0.0)
        robot_v_world = self._extract_robot_velocity(env)
        robot_v_xy = self._xy_from_tensor(robot_v_world[:2]) or (0.0, 0.0)

        ball_pos, ball_lin_vel = self._extract_ball_state(env)
        ball_xy = self._xy_from_tensor(ball_pos[:2]) or (0.0, 0.0)
        ball_v_xy = self._xy_from_tensor(ball_lin_vel[:2]) or (0.0, 0.0)
        target_xy = self._xy_from_tensor(getattr(env, "target_xy", None))

        rew_terms = {}
        if isinstance(infos, dict):
            for k, v in infos.get("rew_terms", {}).items():
                fv = self._to_float(v)
                if fv is not None:
                    rew_terms[k] = fv

        task_terms = self.extract_task_step_metrics(env, infos, rew_terms)
        return StepRecord(
            step=step_idx,
            reward=float(reward),
            robot_xy=robot_xy,
            ball_xy=ball_xy,
            target_xy=target_xy,
            robot_v_xy=robot_v_xy,
            ball_v_xy=ball_v_xy,
            success=bool(infos.get("success", False)),
            fail=bool(infos.get("fail", False)),
            fall=bool(infos.get("fall", False)),
            hit=bool(infos.get("hit", False)),
            rew_terms=rew_terms,
            task=task_terms,
        )

    def _extract_robot_velocity(self, env):
        if hasattr(env, "_world_robot_velocity"):
            return env._world_robot_velocity()
        if hasattr(env, "base_lin_vel"):
            return env.base_lin_vel[0, :3]
        return torch.zeros(3)

    def _extract_ball_state(self, env):
        if hasattr(env, "ball_world"):
            ball_pos, ball_lin_vel, _ = env.ball_world.get_pose(env.root_states)
            return ball_pos, ball_lin_vel
        ball_pos = env.body_states[0, env.controller.num_bodies_robot, 0:3]
        return ball_pos, torch.zeros_like(ball_pos)

    def extract_task_step_metrics(
        self,
        env,
        infos: Dict[str, Any],
        rew_terms: Dict[str, float],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def summarize_task(self, episode: EpisodeRecord) -> Dict[str, float]:
        return {}

    def get_time_series_keys(self) -> Sequence[str]:
        return ()

    def draw_task_overlay(self, ax, episode: EpisodeRecord):
        return None

    def plot_task_dataset(self, axes, episodes: Sequence[EpisodeRecord]):
        return None

    @torch.no_grad()
    def run_episode(self, env, low_model, high_agent, device, episode_index: int) -> EpisodeRecord:
        obs, infos = env.reset()
        obs = obs.to(device)
        steps: List[StepRecord] = []
        ep_return = 0.0

        for step_idx in range(self.max_steps):
            obs_high = self.get_high_level_obs(env).to(device)
            obs_high_np = obs_high.squeeze(0).detach().cpu().numpy()
            action = self.select_high_level_action(high_agent, obs_high_np)
            cmd = self.action_to_command(env, action)
            self.apply_command(env, cmd)
            obs, reward, done_flag, infos = self.rollout_low_level(env, low_model, obs, cmd, device)
            ep_return += float(reward)
            steps.append(self._extract_common_step(env, infos, reward, step_idx))

            if self.should_end_episode(steps[-1], step_idx, done_flag):
                break

        return EpisodeRecord(
            summary=self._build_summary(episode_index, steps, ep_return),
            steps=steps,
        )

    def should_end_episode(self, step_record: StepRecord, step_idx: int, done_flag: bool) -> bool:
        return done_flag or step_record.success or step_record.fail or step_record.fall or (step_idx + 1 >= self.max_steps)

    def _build_summary(self, episode_index: int, steps: Sequence[StepRecord], ep_return: float) -> EpisodeSummary:
        if not steps:
            empty_xy = (0.0, 0.0)
            return EpisodeSummary(
                episode_index=episode_index,
                ep_return=0.0,
                length=0,
                success=False,
                fail=False,
                fall=False,
                hit=False,
                hit_step=None,
                success_step=None,
                fail_step=None,
                final_robot_xy=empty_xy,
                final_ball_xy=empty_xy,
                final_target_xy=None,
                initial_robot_to_ball_dist=None,
                min_robot_to_ball_dist=None,
                final_robot_to_ball_dist=None,
                task_metrics={},
            )

        dists = [
            float(np.linalg.norm(np.asarray(s.robot_xy) - np.asarray(s.ball_xy)))
            for s in steps
        ]
        hit_step = next((s.step for s in steps if s.hit), None)
        success_step = next((s.step for s in steps if s.success), None)
        fail_step = next((s.step for s in steps if s.fail), None)

        episode = EpisodeRecord(summary=None, steps=list(steps))  # type: ignore[arg-type]
        task_metrics = self.summarize_task(episode)
        return EpisodeSummary(
            episode_index=episode_index,
            ep_return=float(ep_return),
            length=len(steps),
            success=any(s.success for s in steps),
            fail=any(s.fail for s in steps),
            fall=any(s.fall for s in steps),
            hit=any(s.hit for s in steps),
            hit_step=hit_step,
            success_step=success_step,
            fail_step=fail_step,
            final_robot_xy=steps[-1].robot_xy,
            final_ball_xy=steps[-1].ball_xy,
            final_target_xy=steps[-1].target_xy,
            initial_robot_to_ball_dist=float(dists[0]) if dists else None,
            min_robot_to_ball_dist=float(min(dists)) if dists else None,
            final_robot_to_ball_dist=float(dists[-1]) if dists else None,
            task_metrics=task_metrics,
        )

    @torch.no_grad()
    def evaluate(
        self,
        env,
        low_model,
        high_agent,
        device,
        episodes: int = 10,
        tb=None,
        global_step: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if hasattr(high_agent, "set_eval_mode"):
            try:
                high_agent.set_eval_mode(True)
            except Exception:
                pass

        records = [
            self.run_episode(env, low_model, high_agent, device, episode_index=i)
            for i in range(int(episodes))
        ]
        aggregate = self._aggregate(records)

        if tb is not None:
            self._write_tb(tb, aggregate)

        out_dir = self._prepare_output_dir(save_dir, global_step)
        if out_dir is not None:
            self.export(records, aggregate, out_dir)

        return {
            "aggregate": aggregate,
            "episodes": [self._episode_to_dict(record) for record in records],
            "output_dir": out_dir,
        }

    def _aggregate(self, records: Sequence[EpisodeRecord]) -> Dict[str, Any]:
        summaries = [r.summary for r in records]
        success_rate = float(np.mean([float(s.success) for s in summaries])) if summaries else 0.0
        fail_rate = float(np.mean([float(s.fail) for s in summaries])) if summaries else 0.0
        fall_rate = float(np.mean([float(s.fall) for s in summaries])) if summaries else 0.0
        avg_return = float(np.mean([s.ep_return for s in summaries])) if summaries else 0.0
        avg_length = float(np.mean([s.length for s in summaries])) if summaries else 0.0
        avg_init_dist = self._mean_optional([s.initial_robot_to_ball_dist for s in summaries])
        avg_min_dist = self._mean_optional([s.min_robot_to_ball_dist for s in summaries])
        avg_final_dist = self._mean_optional([s.final_robot_to_ball_dist for s in summaries])
        return {
            "task": self.task_name,
            "episodes": len(records),
            "success_rate": success_rate,
            "fail_rate": fail_rate,
            "fall_rate": fall_rate,
            "avg_return": avg_return,
            "avg_length": avg_length,
            "avg_initial_robot_to_ball_dist": avg_init_dist,
            "avg_min_robot_to_ball_dist": avg_min_dist,
            "avg_final_robot_to_ball_dist": avg_final_dist,
        }

    def _mean_optional(self, values: Sequence[Optional[float]]) -> Optional[float]:
        valid = [float(v) for v in values if v is not None]
        return float(np.mean(valid)) if valid else None

    def _write_tb(self, tb, aggregate: Dict[str, Any]):
        prefix = self.tb_prefix
        for key, value in aggregate.items():
            if isinstance(value, (int, float)):
                tb.add_scalar(f"{prefix}/{key}", float(value))

    def _prepare_output_dir(self, save_dir: Optional[str], global_step: Optional[int]) -> Optional[str]:
        root = save_dir or self.save_dir
        if root is None:
            return None
        step_str = f"step_{global_step}" if global_step is not None else time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(root, self.task_name, step_str)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def export(self, records: Sequence[EpisodeRecord], aggregate: Dict[str, Any], out_dir: str):
        self._export_json(records, aggregate, out_dir)
        self._export_summary_csv(records, out_dir)
        self._export_key_cases(records, out_dir)
        if self.render_plots:
            self._export_plots(records, out_dir)

    def _episode_to_dict(self, record: EpisodeRecord) -> Dict[str, Any]:
        return {
            "summary": asdict(record.summary),
            "steps": [asdict(step) for step in record.steps],
        }

    def _export_json(self, records: Sequence[EpisodeRecord], aggregate: Dict[str, Any], out_dir: str):
        payload = {
            "aggregate": aggregate,
            "episodes": [self._episode_to_dict(r) for r in records],
        }
        with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _export_summary_csv(self, records: Sequence[EpisodeRecord], out_dir: str):
        path = os.path.join(out_dir, "episode_summary.csv")
        rows = []
        for record in records:
            row = asdict(record.summary)
            task_metrics = row.pop("task_metrics", {})
            for k, v in task_metrics.items():
                row[f"task/{k}"] = v
            rows.append(row)

        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _export_key_cases(self, records: Sequence[EpisodeRecord], out_dir: str):
        selected = self.select_key_episodes(records)
        payload = {
            key: (None if record is None else self._episode_to_dict(record))
            for key, record in selected.items()
        }
        with open(os.path.join(out_dir, "key_cases.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def select_key_episodes(self, records: Sequence[EpisodeRecord]) -> Dict[str, Optional[EpisodeRecord]]:
        if not records:
            return {"best_success": None, "worst_failure": None, "longest": None}

        success_records = [r for r in records if r.summary.success]
        fail_records = [r for r in records if r.summary.fail or r.summary.fall or not r.summary.success]
        return {
            "best_success": max(success_records, key=lambda r: r.summary.ep_return) if success_records else None,
            "worst_failure": min(fail_records, key=lambda r: r.summary.ep_return) if fail_records else None,
            "longest": max(records, key=lambda r: r.summary.length),
        }

    def _export_plots(self, records: Sequence[EpisodeRecord], out_dir: str):
        import matplotlib.pyplot as plt

        episodes_dir = os.path.join(out_dir, "episodes")
        os.makedirs(episodes_dir, exist_ok=True)
        for record in records:
            fig = self.plot_episode(record)
            fig.savefig(os.path.join(episodes_dir, f"episode_{record.summary.episode_index:03d}.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

        overview = self.plot_dataset(records)
        overview.savefig(os.path.join(out_dir, "dataset_overview.png"), dpi=180, bbox_inches="tight")
        plt.close(overview)

    def plot_episode(self, episode: EpisodeRecord):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax0, ax1 = axes
        robot_xy = np.asarray([s.robot_xy for s in episode.steps], dtype=np.float32)
        ball_xy = np.asarray([s.ball_xy for s in episode.steps], dtype=np.float32)

        ax0.plot(robot_xy[:, 0], robot_xy[:, 1], label="robot", color="#0f766e", linewidth=2.0)
        ax0.plot(ball_xy[:, 0], ball_xy[:, 1], label="ball", color="#b45309", linewidth=2.0)
        ax0.scatter(robot_xy[0, 0], robot_xy[0, 1], color="#14b8a6", s=40, marker="o")
        ax0.scatter(ball_xy[0, 0], ball_xy[0, 1], color="#f59e0b", s=40, marker="o")
        ax0.scatter(robot_xy[-1, 0], robot_xy[-1, 1], color="#134e4a", s=50, marker="x")
        ax0.scatter(ball_xy[-1, 0], ball_xy[-1, 1], color="#92400e", s=50, marker="x")
        if episode.steps[-1].target_xy is not None:
            target_xy = np.asarray(episode.steps[-1].target_xy, dtype=np.float32)
            ax0.scatter(target_xy[0], target_xy[1], color="#7c3aed", s=70, marker="*")
        self.draw_task_overlay(ax0, episode)
        ax0.set_title(
            f"{self.task_name} Episode {episode.summary.episode_index} | "
            f"succ={int(episode.summary.success)} fail={int(episode.summary.fail)}"
        )
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.axis("equal")
        ax0.grid(alpha=0.25)
        ax0.legend(loc="best")

        steps = [s.step for s in episode.steps]
        ax1.plot(steps, [s.reward for s in episode.steps], label="reward", color="#111827", linewidth=1.8)
        for key in self.get_time_series_keys():
            series = [s.task.get(key) for s in episode.steps]
            if any(v is not None for v in series):
                ax1.plot(steps, series, label=key, linewidth=1.5)
        ax1.set_title("Time Series")
        ax1.set_xlabel("step")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best")
        return fig

    def plot_dataset(self, records: Sequence[EpisodeRecord]):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax0, ax1 = axes
        returns = [r.summary.ep_return for r in records]
        lengths = [r.summary.length for r in records]
        colors = ["#059669" if r.summary.success else "#dc2626" for r in records]

        ax0.scatter(lengths, returns, c=colors, alpha=0.85)
        ax0.set_xlabel("episode length")
        ax0.set_ylabel("episode return")
        ax0.set_title(f"{self.task_name} Episodes")
        ax0.grid(alpha=0.25)

        final_dists = [
            r.summary.final_robot_to_ball_dist
            for r in records
            if r.summary.final_robot_to_ball_dist is not None
        ]
        if final_dists:
            ax1.hist(final_dists, bins=min(20, max(5, len(final_dists))), color="#2563eb", alpha=0.85)
        ax1.set_xlabel("final robot-ball distance")
        ax1.set_title("Final Distance Distribution")
        ax1.grid(alpha=0.25)

        self.plot_task_dataset(axes, records)
        return fig
