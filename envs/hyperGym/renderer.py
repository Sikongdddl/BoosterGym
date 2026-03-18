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
EGO_MARKER_COLOR = (255, 255, 255)


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
    return render_record_with_camera(
        record=record,
        field_size=field_size,
        frame_size=frame_size,
        camera_mode="global",
    )


def render_record_with_camera(
    record: Dict,
    field_size: Tuple[float, float],
    frame_size: Tuple[int, int] = (960, 640),
    camera_mode: str = "global",
    ego_window_size: Tuple[float, float] = (4.8, 3.6),
) -> Image.Image:
    if camera_mode == "global":
        return _render_global_record(record=record, field_size=field_size, frame_size=frame_size)
    if camera_mode.startswith("ego_fp_"):
        return _render_ego_first_person_record(
            record=record,
            field_size=field_size,
            frame_size=frame_size,
            ego_player_id=camera_mode[len("ego_fp_"):],
        )
    if camera_mode.startswith("ego_"):
        return _render_ego_record(
            record=record,
            field_size=field_size,
            frame_size=frame_size,
            ego_player_id=camera_mode[len("ego_"):],
            ego_window_size=ego_window_size,
        )
    raise ValueError(f"Unknown camera_mode={camera_mode}")


def _render_global_record(
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
    _draw_entities(
        draw=draw,
        state=state,
        field_box=field_box,
        world_window=((0.0, 0.0), (field_w, field_h)),
        font=font,
        ego_player_id="",
    )
    _draw_footer(draw=draw, frame_size=frame_size, record=record, prefix_text="view=global", font=font)
    return image


def _render_ego_record(
    record: Dict,
    field_size: Tuple[float, float],
    frame_size: Tuple[int, int],
    ego_player_id: str,
    ego_window_size: Tuple[float, float],
) -> Image.Image:
    width_px, height_px = frame_size
    panel_h = 128
    margin = 32
    state = record["info"]["state"]
    ego_player = next((player for player in state["players"] if player["player_id"] == ego_player_id), None)
    if ego_player is None:
        raise ValueError(f"Unknown ego_player_id={ego_player_id}")

    ego_window_w = max(float(ego_window_size[0]), 1.0)
    ego_window_h = max(float(ego_window_size[1]), 1.0)
    world_left = float(ego_player["position"][0]) - 0.5 * ego_window_w
    world_right = float(ego_player["position"][0]) + 0.5 * ego_window_w
    world_bottom = float(ego_player["position"][1]) - 0.5 * ego_window_h
    world_top = float(ego_player["position"][1]) + 0.5 * ego_window_h

    field_px_w = width_px - margin * 2
    field_px_h = height_px - panel_h - margin * 2
    field_box = (margin, margin, margin + field_px_w, margin + field_px_h)

    image = Image.new("RGB", frame_size, PANEL_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rounded_rectangle(field_box, radius=18, fill=FIELD_COLOR, outline=LINE_COLOR, width=3)
    _draw_ego_grid(draw, field_box)
    _draw_entities(
        draw=draw,
        state=state,
        field_box=field_box,
        world_window=((world_left, world_bottom), (world_right, world_top)),
        font=font,
        ego_player_id=ego_player_id,
    )
    prefix = (
        f"view=ego:{ego_player_id}  center=({float(ego_player['position'][0]):.2f},{float(ego_player['position'][1]):.2f})"
        f"  window=({ego_window_w:.2f},{ego_window_h:.2f})"
    )
    _draw_footer(draw=draw, frame_size=frame_size, record=record, prefix_text=prefix, font=font)
    return image


def _render_ego_first_person_record(
    record: Dict,
    field_size: Tuple[float, float],
    frame_size: Tuple[int, int],
    ego_player_id: str,
) -> Image.Image:
    width_px, height_px = frame_size
    panel_h = 128
    margin = 32
    state = record["info"]["state"]
    ego_player = next((player for player in state["players"] if player["player_id"] == ego_player_id), None)
    if ego_player is None:
        raise ValueError(f"Unknown ego_player_id={ego_player_id}")

    view_box = (margin, margin, width_px - margin, height_px - panel_h - margin)
    image = Image.new("RGB", frame_size, PANEL_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    _draw_first_person_background(draw, view_box)
    _draw_first_person_field_lines(draw, view_box, state, ego_player)
    _draw_first_person_entities(draw, view_box, state, ego_player, font)
    prefix = f"view=ego_fp:{ego_player_id}"
    _draw_footer(draw=draw, frame_size=frame_size, record=record, prefix_text=prefix, font=font)
    return image


def _to_pixels(position: np.ndarray, field_size: Tuple[float, float], field_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    field_w, field_h = field_size
    left, top, right, bottom = field_box
    x = left + float(position[0]) / max(field_w, 1e-6) * (right - left)
    y = bottom - float(position[1]) / max(field_h, 1e-6) * (bottom - top)
    return int(round(x)), int(round(y))


def _to_pixels_window(
    position: np.ndarray,
    world_window: Tuple[Tuple[float, float], Tuple[float, float]],
    field_box: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    left_w, bottom_w = world_window[0]
    right_w, top_w = world_window[1]
    left_px, top_px, right_px, bottom_px = field_box
    x = left_px + (float(position[0]) - left_w) / max(right_w - left_w, 1e-6) * (right_px - left_px)
    y = bottom_px - (float(position[1]) - bottom_w) / max(top_w - bottom_w, 1e-6) * (bottom_px - top_px)
    return int(round(x)), int(round(y))


def _draw_entities(
    draw: ImageDraw.ImageDraw,
    state: Dict,
    field_box: Tuple[int, int, int, int],
    world_window: Tuple[Tuple[float, float], Tuple[float, float]],
    font: ImageFont.ImageFont,
    ego_player_id: str,
) -> None:
    left_w, bottom_w = world_window[0]
    right_w, top_w = world_window[1]
    for player in state["players"]:
        position = np.asarray(player["position"], dtype=np.float32)
        if not (left_w <= float(position[0]) <= right_w and bottom_w <= float(position[1]) <= top_w):
            continue
        px, py = _to_pixels_window(position, world_window, field_box)
        fill = HOME_COLOR if player["team"] == "home" else AWAY_COLOR
        radius = 14
        outline = EGO_MARKER_COLOR if player["player_id"] == ego_player_id else (255, 255, 255)
        outline_w = 4 if player["player_id"] == ego_player_id else 2
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill, outline=outline, width=outline_w)
        if player["has_ball"]:
            halo = radius + 6
            draw.ellipse((px - halo, py - halo, px + halo, py + halo), outline=BALL_COLOR, width=3)
        label = player["player_id"]
        if player["player_id"] == ego_player_id:
            label += " [ego]"
        draw.text((px + 12, py - 18), label, fill=TEXT_COLOR, font=font)

    ball_position = np.asarray(state["ball_position"], dtype=np.float32)
    if left_w <= float(ball_position[0]) <= right_w and bottom_w <= float(ball_position[1]) <= top_w:
        ball_px, ball_py = _to_pixels_window(ball_position, world_window, field_box)
        ball_r = 7
        draw.ellipse((ball_px - ball_r, ball_py - ball_r, ball_px + ball_r, ball_py + ball_r), fill=BALL_COLOR, outline=(50, 50, 50), width=1)


def _draw_ego_grid(draw: ImageDraw.ImageDraw, field_box: Tuple[int, int, int, int]) -> None:
    left, top, right, bottom = field_box
    width = right - left
    height = bottom - top
    thirds_x = [left + width / 3.0, left + 2.0 * width / 3.0]
    thirds_y = [top + height / 3.0, top + 2.0 * height / 3.0]
    for x in thirds_x:
        draw.line((x, top, x, bottom), fill=(255, 255, 255, 60), width=1)
    for y in thirds_y:
        draw.line((left, y, right, y), fill=(255, 255, 255, 60), width=1)


def _draw_footer(
    draw: ImageDraw.ImageDraw,
    frame_size: Tuple[int, int],
    record: Dict,
    prefix_text: str,
    font: ImageFont.ImageFont,
) -> None:
    width_px, height_px = frame_size
    panel_h = 128
    panel_top = height_px - panel_h
    draw.rectangle((0, panel_top, width_px, height_px), fill=(18, 25, 34))
    state = record["info"]["state"]
    action_text = _format_action(record.get("action"))
    events_text = _format_events(record["info"].get("events", []))
    reward = float(record.get("reward", 0.0))
    done = bool(record.get("done", False))
    step = int(state["step"])
    owner = state["ball_owner_id"] or "free"
    winner = state.get("winner") or "-"
    status = (
        f"{prefix_text}  step={step}  reward={reward:+.3f}  owner={owner}  done={done}"
        f"  goal={bool(state['goal'])}  winner={winner}"
    )
    draw.text((26, panel_top + 18), status, fill=TEXT_COLOR, font=font)
    draw.text((26, panel_top + 44), f"action: {action_text}", fill=(200, 225, 255), font=font)
    away_action_text = _format_action(record.get("away_action"))
    draw.text((26, panel_top + 70), f"away: {away_action_text}", fill=(255, 178, 178), font=font)
    draw.text((26, panel_top + 92), f"events: {events_text}", fill=EVENT_COLOR, font=font)


def _draw_first_person_background(draw: ImageDraw.ImageDraw, view_box: Tuple[int, int, int, int]) -> None:
    left, top, right, bottom = view_box
    horizon = top + int(0.32 * (bottom - top))
    draw.rounded_rectangle(view_box, radius=18, fill=(28, 66, 102), outline=LINE_COLOR, width=3)
    draw.rectangle((left + 2, horizon, right - 2, bottom - 2), fill=FIELD_COLOR)
    draw.rectangle((left + 2, top + 2, right - 2, horizon), fill=(42, 76, 110))
    draw.polygon(
        [
            (left + int(0.18 * (right - left)), bottom),
            (right - int(0.18 * (right - left)), bottom),
            (right - int(0.05 * (right - left)), horizon),
            (left + int(0.05 * (right - left)), horizon),
        ],
        outline=LINE_COLOR,
        fill=(47, 128, 78),
    )
    draw.line((left + int(0.5 * (right - left)), horizon, left + int(0.5 * (right - left)), bottom), fill=(235, 235, 235), width=2)


def _draw_first_person_field_lines(
    draw: ImageDraw.ImageDraw,
    view_box: Tuple[int, int, int, int],
    state: Dict,
    ego_player: Dict,
) -> None:
    left, top, right, bottom = view_box
    segments = [
        ((0.0, 0.0), (0.0, float(state["field_size"][1]))),
        ((float(state["field_size"][0]), 0.0), (float(state["field_size"][0]), float(state["field_size"][1]))),
        ((0.0, 0.0), (float(state["field_size"][0]), 0.0)),
        ((0.0, float(state["field_size"][1])), (float(state["field_size"][0]), float(state["field_size"][1]))),
        ((0.0, 0.5 * float(state["field_size"][1]) - float(state.get("goal_half_width", 1.0))), (0.0, 0.5 * float(state["field_size"][1]) + float(state.get("goal_half_width", 1.0)))),
        ((float(state["field_size"][0]), 0.5 * float(state["field_size"][1]) - float(state.get("goal_half_width", 1.0))), (float(state["field_size"][0]), 0.5 * float(state["field_size"][1]) + float(state.get("goal_half_width", 1.0)))),
    ]
    for start, end in segments:
        points = []
        for alpha in np.linspace(0.0, 1.0, num=18):
            world = np.asarray([
                (1.0 - alpha) * start[0] + alpha * end[0],
                (1.0 - alpha) * start[1] + alpha * end[1],
            ], dtype=np.float32)
            projected = _project_first_person_point(world, ego_player, view_box)
            if projected is not None:
                points.append((projected[0], projected[1]))
        if len(points) >= 2:
            draw.line(points, fill=(240, 240, 240), width=2)


def _draw_first_person_entities(
    draw: ImageDraw.ImageDraw,
    view_box: Tuple[int, int, int, int],
    state: Dict,
    ego_player: Dict,
    font: ImageFont.ImageFont,
) -> None:
    entities = []
    for player in state["players"]:
        projected = _project_first_person_point(np.asarray(player["position"], dtype=np.float32), ego_player, view_box)
        if projected is None:
            continue
        screen_x, screen_y, depth = projected
        entities.append(("player", depth, player, (screen_x, screen_y)))
    ball_projected = _project_first_person_point(np.asarray(state["ball_position"], dtype=np.float32), ego_player, view_box)
    if ball_projected is not None:
        entities.append(("ball", ball_projected[2], state, (ball_projected[0], ball_projected[1])))

    entities.sort(key=lambda item: item[1], reverse=True)
    for kind, depth, payload, (screen_x, screen_y) in entities:
        if kind == "ball":
            radius = int(np.clip(18.0 / max(depth, 0.35), 5, 16))
            draw.ellipse((screen_x - radius, screen_y - radius, screen_x + radius, screen_y + radius), fill=BALL_COLOR, outline=(40, 40, 40), width=1)
            continue

        player = payload
        radius = int(np.clip(28.0 / max(depth, 0.35), 8, 22))
        fill = HOME_COLOR if player["team"] == "home" else AWAY_COLOR
        outline = EGO_MARKER_COLOR if player["player_id"] == ego_player["player_id"] else (255, 255, 255)
        outline_w = 4 if player["player_id"] == ego_player["player_id"] else 2
        draw.ellipse((screen_x - radius, screen_y - radius, screen_x + radius, screen_y + radius), fill=fill, outline=outline, width=outline_w)
        if player["has_ball"]:
            halo = radius + 5
            draw.ellipse((screen_x - halo, screen_y - halo, screen_x + halo, screen_y + halo), outline=BALL_COLOR, width=3)
        label = player["player_id"]
        if player["player_id"] == ego_player["player_id"]:
            label += " [ego]"
        draw.text((screen_x + radius + 4, screen_y - radius), label, fill=TEXT_COLOR, font=font)


def _project_first_person_point(
    world_point: np.ndarray,
    ego_player: Dict,
    view_box: Tuple[int, int, int, int],
) -> Tuple[int, int, float] | None:
    ego_pos = np.asarray(ego_player["position"], dtype=np.float32)
    heading = float(ego_player.get("heading", 0.0))
    forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
    right = np.asarray([forward[1], -forward[0]], dtype=np.float32)
    relative = np.asarray(world_point, dtype=np.float32) - ego_pos
    depth = float(np.dot(relative, forward))
    lateral = float(np.dot(relative, right))

    near_clip = -0.25
    far_clip = 6.5
    if depth < near_clip or depth > far_clip:
        return None

    left, top, right_px, bottom = view_box
    horizon = top + int(0.32 * (bottom - top))
    ground_bottom = bottom - 10
    half_width = 0.5 * (right_px - left)
    center_x = left + int(half_width)
    effective_depth = max(depth + 0.75, 0.35)
    fov_scale = 1.15
    x = center_x + lateral / effective_depth * half_width / fov_scale
    if x < left - 40 or x > right_px + 40:
        return None

    depth_ratio = np.clip((depth + 0.25) / (far_clip + 0.25), 0.0, 1.0)
    y = ground_bottom - (depth_ratio ** 0.78) * (ground_bottom - horizon)
    return int(round(x)), int(round(y)), effective_depth


def _format_action(action: Dict | None) -> str:
    if not action:
        return "reset"
    if "players" in action:
        action = action["players"]
    if any(isinstance(value, dict) and "skill" in value for value in action.values()):
        parts: List[str] = []
        for player_id, player_action in action.items():
            skill = player_action.get("skill", player_action.get("type", "move"))
            target = np.asarray(player_action.get("target", [0.0, 0.0]), dtype=np.float32)
            parts.append(f"{player_id}:{skill}({target[0]:.2f},{target[1]:.2f})")
        return " | ".join(parts)
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
