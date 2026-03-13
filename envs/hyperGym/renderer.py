from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HOME_COLOR = (65, 127, 255)
AWAY_COLOR = (255, 92, 92)
BALL_COLOR = (255, 236, 120)
FIELD_COLOR = (43, 122, 72)
LINE_COLOR = (245, 245, 245)
TEXT_COLOR = (248, 248, 248)
PANEL_COLOR = (14, 20, 27)
EVENT_COLOR = (255, 204, 102)


def render_episode_mp4(
    episode: Iterable[Dict],
    output_path: str | Path,
    field_size: Tuple[float, float],
    fps: int = 8,
    frame_size: Tuple[int, int] = (960, 640),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p") as writer:
        for record in episode:
            frame = render_record(record, field_size=field_size, frame_size=frame_size)
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return output


def render_record(
    record: Dict,
    field_size: Tuple[float, float],
    frame_size: Tuple[int, int] = (960, 640),
) -> Image.Image:
    width_px, height_px = frame_size
    field_w, field_h = field_size
    panel_h = 128
    margin = 32
    scale = min((width_px - margin * 2) / max(field_w, 1e-6), (height_px - panel_h - margin * 2) / max(field_h, 1e-6))
    field_px_w = int(round(field_w * scale))
    field_px_h = int(round(field_h * scale))
    origin_x = (width_px - field_px_w) // 2
    origin_y = margin
    field_box = (origin_x, origin_y, origin_x + field_px_w, origin_y + field_px_h)

    image = Image.new("RGB", frame_size, PANEL_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rounded_rectangle(field_box, radius=18, fill=FIELD_COLOR, outline=LINE_COLOR, width=3)
    mid_x = origin_x + field_px_w / 2
    draw.line((mid_x, origin_y, mid_x, origin_y + field_px_h), fill=LINE_COLOR, width=2)
    center_r = int(max(18, 0.12 * scale))
    draw.ellipse((mid_x - center_r, origin_y + field_px_h / 2 - center_r, mid_x + center_r, origin_y + field_px_h / 2 + center_r), outline=LINE_COLOR, width=2)

    state = record["info"]["state"]
    goal_half_width = float(state.get("goal_half_width", 1.0))
    goal_center_y = origin_y + field_px_h / 2
    goal_half_h = int(round(goal_half_width * scale))
    draw.line((origin_x, goal_center_y - goal_half_h, origin_x, goal_center_y + goal_half_h), fill=(120, 200, 255), width=5)
    draw.line((origin_x + field_px_w, goal_center_y - goal_half_h, origin_x + field_px_w, goal_center_y + goal_half_h), fill=(255, 215, 110), width=5)
    for player in state["players"]:
        px, py = _to_pixels(player["position"], field_size, field_box)
        fill = HOME_COLOR if player["team"] == "home" else AWAY_COLOR
        radius = 14
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill, outline=(255, 255, 255), width=2)
        if player["has_ball"]:
            halo = radius + 6
            draw.ellipse((px - halo, py - halo, px + halo, py + halo), outline=BALL_COLOR, width=3)
        draw.text((px + 12, py - 18), player["player_id"], fill=TEXT_COLOR, font=font)

    ball_px, ball_py = _to_pixels(state["ball_position"], field_size, field_box)
    ball_r = 7
    draw.ellipse((ball_px - ball_r, ball_py - ball_r, ball_px + ball_r, ball_py + ball_r), fill=BALL_COLOR, outline=(50, 50, 50), width=1)

    panel_top = height_px - panel_h
    draw.rectangle((0, panel_top, width_px, height_px), fill=(18, 25, 34))
    action_text = _format_action(record.get("action"))
    events_text = _format_events(record["info"].get("events", []))
    reward = float(record.get("reward", 0.0))
    done = bool(record.get("done", False))
    step = int(state["step"])
    owner = state["ball_owner_id"] or "free"
    winner = state.get("winner") or "-"
    status = f"step={step}  reward={reward:+.3f}  owner={owner}  done={done}  goal={bool(state['goal'])}  winner={winner}"
    draw.text((26, panel_top + 18), status, fill=TEXT_COLOR, font=font)
    draw.text((26, panel_top + 44), f"action: {action_text}", fill=(200, 225, 255), font=font)
    away_action_text = _format_action(record.get("away_action"))
    draw.text((26, panel_top + 70), f"away: {away_action_text}", fill=(255, 178, 178), font=font)
    draw.text((26, panel_top + 92), f"events: {events_text}", fill=EVENT_COLOR, font=font)

    return image


def _to_pixels(position: np.ndarray, field_size: Tuple[float, float], field_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    field_w, field_h = field_size
    left, top, right, bottom = field_box
    x = left + float(position[0]) / max(field_w, 1e-6) * (right - left)
    y = bottom - float(position[1]) / max(field_h, 1e-6) * (bottom - top)
    return int(round(x)), int(round(y))


def _format_action(action: Dict | None) -> str:
    if not action:
        return "reset"
    skill = action.get("skill", action.get("type", "move"))
    target = np.asarray(action.get("target", [0.0, 0.0]), dtype=np.float32)
    return f"{skill} -> ({target[0]:.2f}, {target[1]:.2f})"


def _format_events(events: List[Dict]) -> str:
    if not events:
        return "none"
    parts: List[str] = []
    for event in events:
        event_type = event.get("event_type", "unknown")
        payload = ", ".join(f"{key}={value}" for key, value in event.items() if key != "event_type")
        parts.append(f"{event_type}({payload})" if payload else event_type)
    return " | ".join(parts)
