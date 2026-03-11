from typing import Dict, Sequence

import matplotlib.patches as patches
import numpy as np

from eval.evaluator import EpisodeRecord, RLEvaluator


class PassBallEvaluator(RLEvaluator):
    @property
    def task_name(self) -> str:
        return "passBall"

    def extract_task_step_metrics(self, env, infos: Dict, rew_terms: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for key in (
            "ball_dist",
            "ball_speed",
            "ball_align_cos",
            "ball_align_reward",
            "robot_speed",
            "robot_align_cos",
            "robot_align_reward",
            "robot_to_ball_dist",
            "line_cos",
            "line_reward",
            "line_dist_gate",
            "approach_cos",
            "approach_reward",
        ):
            if key in rew_terms:
                metrics[key] = rew_terms[key]
        return metrics

    def summarize_task(self, episode: EpisodeRecord) -> Dict[str, float]:
        ball_dist = [step.task.get("ball_dist") for step in episode.steps if "ball_dist" in step.task]
        ball_speed = [step.task.get("ball_speed") for step in episode.steps if "ball_speed" in step.task]
        line_reward = [step.task.get("line_reward") for step in episode.steps if "line_reward" in step.task]
        return {
            "min_ball_to_target_dist": float(np.min(ball_dist)) if ball_dist else np.nan,
            "final_ball_to_target_dist": float(ball_dist[-1]) if ball_dist else np.nan,
            "max_ball_speed": float(np.max(ball_speed)) if ball_speed else np.nan,
            "avg_line_reward": float(np.mean(line_reward)) if line_reward else np.nan,
        }

    def get_time_series_keys(self) -> Sequence[str]:
        return ("ball_dist", "ball_speed", "robot_to_ball_dist", "line_reward", "approach_reward")

    def draw_task_overlay(self, ax, episode: EpisodeRecord):
        if episode.steps[-1].target_xy is None:
            return
        target_xy = episode.steps[-1].target_xy
        ax.add_patch(
            patches.Circle(
                target_xy,
                radius=0.6,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="#7c3aed",
                alpha=0.9,
            )
        )

