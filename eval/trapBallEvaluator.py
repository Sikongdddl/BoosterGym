from typing import Dict, Sequence

import matplotlib.lines as mlines
import numpy as np

from eval.evaluator import EpisodeRecord, RLEvaluator


class TrapBallEvaluator(RLEvaluator):
    @property
    def task_name(self) -> str:
        return "trapBall"

    def extract_task_step_metrics(self, env, infos: Dict, rew_terms: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for key in (
            "ball_speed",
            "robot_speed",
            "robot_to_ball_dist",
            "intercept_dist",
            "signed_line_dist",
            "lateral_offset",
            "approach_speed",
            "approach_cos",
            "intercept_cos",
            "crossed_line",
        ):
            if key in rew_terms:
                metrics[key] = rew_terms[key]
        return metrics

    def summarize_task(self, episode: EpisodeRecord) -> Dict[str, float]:
        intercept_dist = [step.task.get("intercept_dist") for step in episode.steps if "intercept_dist" in step.task]
        line_dist = [step.task.get("signed_line_dist") for step in episode.steps if "signed_line_dist" in step.task]
        return {
            "min_intercept_dist": float(np.min(intercept_dist)) if intercept_dist else np.nan,
            "final_signed_line_dist": float(line_dist[-1]) if line_dist else np.nan,
            "min_signed_line_dist": float(np.min(line_dist)) if line_dist else np.nan,
        }

    def get_time_series_keys(self) -> Sequence[str]:
        return ("robot_to_ball_dist", "intercept_dist", "signed_line_dist", "approach_cos", "intercept_cos")

    def draw_task_overlay(self, ax, episode: EpisodeRecord):
        if not episode.steps:
            return
        first = episode.steps[0]
        last = episode.steps[-1]
        start = np.asarray(first.robot_xy, dtype=np.float32)
        if last.target_xy is not None:
            ax.scatter(last.target_xy[0], last.target_xy[1], color="#7c3aed", s=70, marker="*")

        x0 = float(start[0])
        y0 = float(start[1])
        line = mlines.Line2D([x0, x0], [y0 - 2.5, y0 + 2.5], color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.add_line(line)

