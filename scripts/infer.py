# dumb infer codes for checking effects of chaseBall
# without isaacgym real env, obs are made up
import sys
import os
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.agents.sac.agent import SACAgent

# args fixed
STATE_DIM = 8
ACTION_DIM = 3

def _get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_sac_agent(weights_path = "./scripts/1.pt",
                   device = None) -> SACAgent:
    """
    从 torch.save 保存的 ckpt 中读取完整 agent 对象。
    """
    if device is None:
        device = _get_default_device()

    ckpt = torch.load(weights_path, map_location=device)

    if not (isinstance(ckpt, dict) and "agent" in ckpt):
        raise ValueError(
            f"{weights_path} 不是预期的 ckpt 格式，应包含键 'agent'。"
        )

    agent: SACAgent = ckpt["agent"]

    # 覆盖 device，并把子网络搬到对应 device + eval 模式
    agent.device = device
    for name in ["policy", "q1", "q2", "q1_target", "q2_target"]:
        net = getattr(agent, name, None)
        if net is not None:
            net.to(device)
            net.eval()

    print(f"[Infer] Loaded SAC agent from {weights_path} on {device}")
    # 此时 agent.action_low / action_high 已经是训练时那组范围：
    # vx ∈ [0.0, 0.6], vy ∈ [-0.35, 0.35], yaw ∈ [-1.0, 1.0]
    return agent


@torch.no_grad()
def infer_action(agent: SACAgent, obs_vec, eval_mode: bool = True):
    """
    对单个 8 维观测做推理，返回高层动作。

    参数
    ----
    agent    : 已经 load 好的 SACAgent
    obs_vec  : 长度为 8 的向量（list/tuple/np.ndarray/torch.Tensor 都行）
               语义为：
                 0-1: delta_xy_body
                 2  : dist_xy
                 3-4: cos(bearing), sin(bearing)
                 5-6: v_body_xy
                 7  : speed_toward
    eval_mode: True 时用均值动作（更稳定），False 时带探索噪声（训练风格）

    返回
    ----
    action : np.ndarray, shape = (3,)
             [vx, vy, yaw]，已经过 _scale_action 缩放到了物理范围
    """
    obs_np = np.asarray(obs_vec, dtype=np.float32)

    if obs_np.shape != (STATE_DIM,):
        raise ValueError(
            f"obs_vec 形状必须是 ({STATE_DIM},)，当前是 {obs_np.shape}"
        )

    # SACAgent.select_action 内部会：
    #   1) s -> (1, state_dim)
    #   2) policy.sample()
    #   3) tanh 动作缩放到物理范围
    action = agent.select_action(obs_np, eval_mode=eval_mode)
    # 已经是 numpy，shape=(3,)
    return action


def build_demo_obs():
    """
    构造一个“合理但瞎编”的 demo 观测，方便你测试推理流程。
    语义大概如下：
      - 目标在自车系 x 正方向 1m 处（正前方）
      - dist_xy = 1.0
      - bearing = 0 -> cos=1, sin=0
      - 机器人当前几乎静止
      - speed_toward ~ 0
    """
    delta_x = 1.0
    delta_y = 0.0
    dist_xy = 1.0
    cos_b = 1.0
    sin_b = 0.0
    v_x = 0.0
    v_y = 0.0
    speed_toward = 0.0

    return np.array(
        [
            delta_x,
            delta_y,
            dist_xy,
            cos_b,
            sin_b,
            v_x,
            v_y,
            speed_toward,
        ],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="SAC high-level policy inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="./scripts/1.pt",
        help="权重文件路径（可以是完整 ckpt 或 SACAgent.save() 的结果）",
    )
    parser.add_argument(
        "--input",
        type=float,
        nargs=STATE_DIM,
        help=f"手动输入 {STATE_DIM} 个浮点数作为观测，"
             f"例如：--input 1 0 1 1 0 0 0 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="推理使用的设备，默认自动检测",
    )
    parser.add_argument(
        "--train-mode",
        action="store_true",
        help="使用带随机性的动作（eval_mode=False），默认是均值动作 eval_mode=True",
    )

    args = parser.parse_args()

    device = args.device or _get_default_device()

    # 这里用默认 [-1,1] 动作范围；如果你训练时动作范围不同，可以在这里改
    agent = load_sac_agent(weights_path=args.weights, device=device)

    if args.input is None:
        obs = build_demo_obs()
        print("没有提供 --input，使用 demo 观测：")
        print(obs)
    else:
        obs = np.array(args.input, dtype=np.float32)

    action = infer_action(agent, obs_vec=obs, eval_mode=not args.train_mode)

    print("\n=== 推理结果 ===")
    print("obs (8-dim):", obs)
    print("action (vx, vy, yaw):", action)


if __name__ == "__main__":
    main()