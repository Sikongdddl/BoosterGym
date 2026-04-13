from typing import Dict, Sequence

import matplotlib.patches as patches
import numpy as np

from eval.evaluator import EpisodeRecord, RLEvaluator


class ChaseBallEvaluator(RLEvaluator):
    @property
    def task_name(self) -> str:
        return "chaseBall"

    def extract_task_step_metrics(self, env, infos: Dict, rew_terms: Dict[str, float]) -> Dict[str, float]:
        metrics = {
            "dist_xy": rew_terms.get("dist_xy", float(np.linalg.norm(np.asarray(env.base_pos[0, :2].detach().cpu()) - np.asarray(env.target_xy.detach().cpu())))),
        }
        for key in ("heading_cos", "heading_term", "progress_gain", "speed_toward", "speed_orth", "spin_penalty"):
            if key in rew_terms:
                metrics[key] = rew_terms[key]
        return metrics

    def summarize_task(self, episode: EpisodeRecord) -> Dict[str, float]:
        dists = [step.task.get("dist_xy") for step in episode.steps if "dist_xy" in step.task]
        heading = [step.task.get("heading_cos") for step in episode.steps if "heading_cos" in step.task]
        return {
            "min_dist_xy": float(np.min(dists)) if dists else np.nan,
            "avg_dist_xy": float(np.mean(dists)) if dists else np.nan,
            "avg_heading_cos": float(np.mean(heading)) if heading else np.nan,
        }

    def get_time_series_keys(self) -> Sequence[str]:
        return ("dist_xy", "heading_cos", "progress_gain", "speed_toward")

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
                edgecolor="#2563eb",
                alpha=0.8,
            )
        )

