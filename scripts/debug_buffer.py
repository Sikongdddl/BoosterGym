#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyse_replay.py

用来分析 & 可视化 ReplayBuffer 保存的 npz 文件。

使用方式：
    python analyse_replay.py logs/replay/replay_step_200_N1234.npz

你可以在脚本顶部修改若干索引，适配你当前的 obs 定义。
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# ================== 根据你的 obs 定义配置下标 ==================
# 按你刚才的设想：
# obs = [ robot_x, robot_y,
#         ball_x,  ball_y,
#         v_x,     v_y,
#         target_x,target_y, ...]
ROBOT_XY_IDX = (0, 1)
BALL_XY_IDX  = (2, 3)
VEL_XY_IDX   = (4, 5)
TARGET_XY_IDX = (6, 7)   # 如果 obs 维度 < 8，会自动跳过用不到的分析


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str, help="ReplayBuffer 保存的 npz 文件路径")
    parser.add_argument("--max_transitions", type=int, default=None,
                        help="最多分析多少条 transition（默认全部）")
    parser.add_argument("--show", action="store_true",
                        help="显示图像（不加的话只打印统计信息）")
    return parser.parse_args()


def load_replay(npz_path, max_transitions=None):
    data = np.load(npz_path, allow_pickle=True)
    required_keys = ["s", "a", "r", "s2", "d"]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"npz 文件缺少关键字段 '{k}'，实际 keys = {list(data.keys())}")

    s  = data["s"]
    a  = data["a"]
    r  = data["r"]
    s2 = data["s2"]
    d  = data["d"]
    notes = data["notes"] if "notes" in data else None

    N = s.shape[0]
    if max_transitions is not None and max_transitions < N:
        s  = s[:max_transitions]
        a  = a[:max_transitions]
        r  = r[:max_transitions]
        s2 = s2[:max_transitions]
        d  = d[:max_transitions]
        if notes is not None:
            notes = notes[:max_transitions]
        N  = max_transitions

    return s, a, r, s2, d, notes


def analyse_basic(s, a, r, d, notes=None):
    N = s.shape[0]
    obs_dim = s.shape[1]
    act_shape = a.shape[1:] if a.ndim > 1 else (1,)

    print("========== 基本信息 ==========")
    print(f"Transitions:      {N}")
    print(f"Obs dim:          {obs_dim}")
    print(f"Action shape:     {act_shape}")
    print(f"Done True count:  {int(d.sum())}")
    if notes is not None:
        print(f"Notes shape:      {notes.shape}")
        print(f"First 3 notes:    {[str(notes[i]) for i in range(min(3, len(notes)))]}")

    print("\n========== Buffer 全部记录 ==========")
    for i in range(N):
        print(f"[{i}] r={r[i]:.4f} done={d[i]} note={str(notes[i]) if notes is not None else ''}")

    print("\n========== 奖励统计 ==========")
    print(f"r mean: {r.mean():.4f}")
    print(f"r std : {r.std():.4f}")
    print(f"r min : {r.min():.4f}")
    print(f"r max : {r.max():.4f}")

    # ---------- 按 episode 分割 ----------
    episodes = []
    ep_start = 0
    for i in range(N):
        terminal = bool(d[i]) or (i == N - 1)
        if terminal:
            ep_len = i - ep_start + 1
            ep_ret = r[ep_start:i+1].sum()
            episodes.append((ep_len, ep_ret))
            ep_start = i + 1

    if len(episodes) == 0:
        print("\n[Warning] 没有检测到 episode（done 全 False？）")
        return {}

    ep_lens  = np.array([e[0] for e in episodes])
    ep_rets  = np.array([e[1] for e in episodes])

    print("\n========== Episode 统计 ==========")
    print(f"Episodes:         {len(episodes)}")
    print(f"Ep length mean:   {ep_lens.mean():.2f} (min {ep_lens.min()}, max {ep_lens.max()})")
    print(f"Ep return mean:   {ep_rets.mean():.3f} (min {ep_rets.min():.3f}, max {ep_rets.max():.3f})")

    stats = {
        "ep_lens": ep_lens,
        "ep_rets": ep_rets,
    }
    return stats


def plot_rewards(r, stats):
    N = len(r)

    plt.figure(figsize=(10, 4))
    plt.plot(r)
    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.title("Reward Timeseries")

    plt.figure(figsize=(5, 4))
    plt.hist(r, bins=50)
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.title("Reward Histogram")

    if stats:
        ep_lens = stats["ep_lens"]
        ep_rets = stats["ep_rets"]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(ep_lens, bins=30)
        plt.xlabel("episode length")
        plt.ylabel("count")
        plt.title("Episode Length Distribution")

        plt.subplot(1, 2, 2)
        plt.hist(ep_rets, bins=30)
        plt.xlabel("episode return")
        plt.ylabel("count")
        plt.title("Episode Return Distribution")

        plt.tight_layout()


def plot_geometry(s):
    """基于你当前的 obs 结构，做一些几何可视化：机器人 / 球 / target 的关系。"""
    N, obs_dim = s.shape

    if obs_dim < 4:
        print("\n[Geom] obs 维度 < 4，无法解析 robot/ball xy，跳过几何可视化。")
        return

    robot_xy = s[:, ROBOT_XY_IDX[0]:ROBOT_XY_IDX[1]+1]
    ball_xy  = s[:, BALL_XY_IDX[0]:BALL_XY_IDX[1]+1]

    # 计算距离
    dist_rb = np.linalg.norm(ball_xy - robot_xy, axis=1)

    if obs_dim >= 8:
        target_xy = s[:, TARGET_XY_IDX[0]:TARGET_XY_IDX[1]+1]
        dist_bt = np.linalg.norm(target_xy - ball_xy, axis=1)
    else:
        target_xy = None
        dist_bt = None

    # 1) 距离随时间变化
    plt.figure(figsize=(10, 4))
    plt.plot(dist_rb, label="dist(robot, ball)")
    if dist_bt is not None:
        plt.plot(dist_bt, label="dist(ball, target)")
    plt.xlabel("timestep")
    plt.ylabel("distance")
    plt.title("Distances over time")
    plt.legend()

    # 2) XY 平面散点（抽样一些点避免太密）
    idx = np.arange(N)
    if N > 5000:
        idx = np.linspace(0, N - 1, 5000).astype(int)

    plt.figure(figsize=(6, 6))
    plt.scatter(robot_xy[idx, 0], robot_xy[idx, 1], s=2, alpha=0.5, label="robot")
    plt.scatter(ball_xy[idx, 0],  ball_xy[idx, 1],  s=2, alpha=0.5, label="ball")
    if target_xy is not None:
        plt.scatter(target_xy[idx, 0], target_xy[idx, 1], s=2, alpha=0.5, label="target")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY positions (subsampled)")
    plt.legend()
    plt.axis("equal")  # 不强求，但看轨迹会舒服点


def plot_actions(a):
    """简单看一下动作分布，帮助判断策略是否塌缩、是否在用满 action 空间。"""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]

    N, act_dim = a.shape
    print(f"\n[Actions] N={N}, act_dim={act_dim}")

    plt.figure(figsize=(10, 3 * act_dim))
    for i in range(act_dim):
        plt.subplot(act_dim, 1, i + 1)
        plt.hist(a[:, i], bins=50)
        plt.xlabel(f"action[{i}]")
        plt.ylabel("count")
        plt.title(f"Action dim {i} histogram")

    plt.tight_layout()


def main():
    args = parse_args()
    npz_path = args.npz_path

    if not os.path.isfile(npz_path):
        print(f"[Error] 文件不存在: {npz_path}")
        return

    print(f"[Info] 加载 {npz_path}")
    s, a, r, s2, d, notes = load_replay(npz_path, args.max_transitions)

    stats = analyse_basic(s, a, r, d, notes)

    # 只输出基本信息，不画图
    print("\n[Info] 分析完毕。")

if __name__ == "__main__":
    main()
