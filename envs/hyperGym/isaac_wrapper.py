"""
Map IsaacGym sim state into HyperGym-style ``state`` / ``record`` dicts.

Used for the fallback path: run HyperGym policies (or ``render_record``) using
symbolic state projected from IsaacGym tensors, without parsing RGB video.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _clip_to_field(xy: np.ndarray, field_size: Tuple[float, float]) -> np.ndarray:
    w, h = field_size
    out = np.asarray(xy, dtype=np.float32).copy()
    out[0] = float(np.clip(out[0], 0.0, w))
    out[1] = float(np.clip(out[1], 0.0, h))
    return out


def _map_linear(
    xy: np.ndarray,
    bounds_x: Tuple[float, float],
    bounds_y: Tuple[float, float],
    field_size: Tuple[float, float],
) -> np.ndarray:
    """Map world (x,y) from axis-aligned bounds into [0, field_w] x [0, field_h]."""
    fx, fy = field_size
    xmin, xmax = bounds_x
    ymin, ymax = bounds_y
    x = float(xy[0])
    y = float(xy[1])
    tx = 0.0 if xmax <= xmin else (x - xmin) / (xmax - xmin)
    ty = 0.0 if ymax <= ymin else (y - ymin) / (ymax - ymin)
    tx = float(np.clip(tx, 0.0, 1.0))
    ty = float(np.clip(ty, 0.0, 1.0))
    return np.asarray([tx * fx, ty * fy], dtype=np.float32)


def isaac_single_robot_ball_to_hyper_state(
    robot_xy: np.ndarray,
    ball_xy: np.ndarray,
    ball_vel_xy: np.ndarray,
    *,
    field_size: Tuple[float, float] = (10.0, 6.0),
    world_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    step: int = 0,
    ball_owner_id: Optional[str] = None,
    goal_half_width: float = 1.0,
    away_anchor_xy: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build HyperGym-like state from one robot + ball (e.g. PassBall / ChaseBall).

    If ``world_bounds`` is None, uses raw XY clipped to ``field_size`` (assume same scale).
    Otherwise maps (x,y) from bounds to the field rectangle.
    """
    robot_xy = np.asarray(robot_xy, dtype=np.float32).reshape(2)
    ball_xy = np.asarray(ball_xy, dtype=np.float32).reshape(2)
    ball_vel_xy = np.asarray(ball_vel_xy, dtype=np.float32).reshape(2)

    if world_bounds is not None:
        (xmin, xmax), (ymin, ymax) = world_bounds
        r2 = _map_linear(robot_xy, (xmin, xmax), (ymin, ymax), field_size)
        b2 = _map_linear(ball_xy, (xmin, xmax), (ymin, ymax), field_size)
    else:
        r2 = _clip_to_field(robot_xy, field_size)
        b2 = _clip_to_field(ball_xy, field_size)

    if away_anchor_xy is None:
        away_xy = np.asarray([field_size[0] * 0.85, field_size[1] * 0.5], dtype=np.float32)
    else:
        away_xy = _clip_to_field(np.asarray(away_anchor_xy, dtype=np.float32), field_size)

    players: List[Dict[str, Any]] = [
        {
            "player_id": "home_0",
            "team": "home",
            "position": r2.copy(),
            "velocity": np.zeros(2, dtype=np.float32),
            "has_ball": ball_owner_id == "home_0",
            "heading": 0.0,
        },
        {
            "player_id": "away_0",
            "team": "away",
            "position": away_xy.copy(),
            "velocity": np.zeros(2, dtype=np.float32),
            "has_ball": ball_owner_id == "away_0",
            "heading": float(np.pi),
        },
    ]

    return {
        "step": int(step),
        "field_size": np.asarray(field_size, dtype=np.float32),
        "goal_half_width": float(goal_half_width),
        "ball_position": b2.copy(),
        "ball_velocity": ball_vel_xy.copy(),
        "ball_owner_id": ball_owner_id,
        "ball_dynamics": {},
        "goal": False,
        "winner": None,
        "players": players,
    }


def isaac_multi_agent_ball_to_hyper_state(
    player_positions_xy: Sequence[np.ndarray],
    player_ids: Sequence[str],
    teams: Sequence[str],
    ball_xy: np.ndarray,
    ball_vel_xy: np.ndarray,
    *,
    field_size: Tuple[float, float] = (10.0, 6.0),
    world_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    step: int = 0,
    ball_owner_id: Optional[str] = None,
    goal_half_width: float = 1.0,
    headings: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Build HyperGym-like state from N robots (e.g. BoosterT12v2) + ball."""
    ball_xy = np.asarray(ball_xy, dtype=np.float32).reshape(2)
    ball_vel_xy = np.asarray(ball_vel_xy, dtype=np.float32).reshape(2)

    if world_bounds is not None:
        (xmin, xmax), (ymin, ymax) = world_bounds
        b2 = _map_linear(ball_xy, (xmin, xmax), (ymin, ymax), field_size)
    else:
        b2 = _clip_to_field(ball_xy, field_size)

    players: List[Dict[str, Any]] = []
    for i, (pid, team) in enumerate(zip(player_ids, teams)):
        pxy = np.asarray(player_positions_xy[i], dtype=np.float32).reshape(2)
        if world_bounds is not None:
            (xmin, xmax), (ymin, ymax) = world_bounds
            p2 = _map_linear(pxy, (xmin, xmax), (ymin, ymax), field_size)
        else:
            p2 = _clip_to_field(pxy, field_size)
        hd = 0.0 if headings is None else float(headings[i])
        players.append(
            {
                "player_id": str(pid),
                "team": str(team),
                "position": p2,
                "velocity": np.zeros(2, dtype=np.float32),
                "has_ball": ball_owner_id == pid,
                "heading": hd,
            }
        )

    return {
        "step": int(step),
        "field_size": np.asarray(field_size, dtype=np.float32),
        "goal_half_width": float(goal_half_width),
        "ball_position": b2.copy(),
        "ball_velocity": ball_vel_xy.copy(),
        "ball_owner_id": ball_owner_id,
        "ball_dynamics": {},
        "goal": False,
        "winner": None,
        "players": players,
    }


def hyper_state_to_record(
    state: Dict[str, Any],
    *,
    reward: float = 0.0,
    done: bool = False,
    action: Optional[Dict[str, Any]] = None,
    away_action: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Wrap a HyperGym ``state`` dict into the ``record`` shape expected by ``render_record``."""
    return {
        "obs": None,
        "reward": float(reward),
        "done": bool(done),
        "action": action,
        "away_action": away_action,
        "info": {
            "events": list(events or []),
            "state": state,
        },
    }


def isaac_pass_ball_tensors_to_hyper_state(
    root_states: torch.Tensor,
    *,
    field_size: Tuple[float, float] = (10.0, 6.0),
    world_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    step: int = 0,
    ball_owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience: actor 0 = robot base, actor 1 = ball (PassBall / ChaseBall layout)."""
    rs = root_states
    robot_xy = rs[0, 0:2].detach().cpu().numpy()
    ball_xy = rs[1, 0:2].detach().cpu().numpy()
    ball_vel_xy = rs[1, 7:9].detach().cpu().numpy()
    return isaac_single_robot_ball_to_hyper_state(
        robot_xy,
        ball_xy,
        ball_vel_xy,
        field_size=field_size,
        world_bounds=world_bounds,
        step=step,
        ball_owner_id=ball_owner_id,
    )


def isaac_booster_multi_tensors_to_hyper_state(
    root_states: torch.Tensor,
    num_players: int,
    ball_actor_index: int,
    player_layout: Sequence[Dict[str, Any]],
    *,
    field_size: Tuple[float, float] = (10.0, 6.0),
    world_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    step: int = 0,
    ball_owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience: first ``num_players`` rows are robots, ball at ``ball_actor_index``."""
    positions = [root_states[i, 0:2].detach().cpu().numpy() for i in range(num_players)]
    ids = [p["name"] for p in player_layout]
    teams = [p["team"] for p in player_layout]
    headings = [float(p.get("yaw", 0.0)) for p in player_layout]
    ball_xy = root_states[ball_actor_index, 0:2].detach().cpu().numpy()
    ball_vel_xy = root_states[ball_actor_index, 7:9].detach().cpu().numpy()
    return isaac_multi_agent_ball_to_hyper_state(
        positions,
        ids,
        teams,
        ball_xy,
        ball_vel_xy,
        field_size=field_size,
        world_bounds=world_bounds,
        step=step,
        ball_owner_id=ball_owner_id,
        headings=headings,
    )
