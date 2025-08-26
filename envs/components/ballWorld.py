# envs/components/ball_world.py
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate
class BallWorld:
    """
    负责球体的生命周期管理：读取/写入位姿、环带重置、远处重刷。
    依赖于 LowLevelController 暴露的：
      - controller.gym
      - controller.sim
      - controller.num_bodies_robot（球的刚体索引 = 机器人刚体数）
      - （可选）controller.ball_handle 由 Controller 在 _create_envs 中创建
    """
    def __init__(self, controller, *, default_z: float = 0.12):
        self.controller = controller
        # 球在 body_states 里的索引：紧随机器人刚体之后
        self.ball_idx = int(controller.num_bodies_robot)
        self.default_z = float(default_z)

        # 兼容：若控制器里已经创建了球（推荐做法），记住 handle；没有也不会影响后续 root-state 写入
        self.ball_handle = getattr(controller, "ball_handle", None)

    # ------ 读/写便捷方法 ------
    def get_index(self) -> int:
        return self.ball_idx

    def get_pose(self, root_states: torch.Tensor):
        """
        返回 (pos[3], lin_vel[3], ang_vel[3])，基于 Isaac Gym actor_root_state_tensor 切片。
        root_states shape: [num_actors, 13]
        """
        x = root_states[1, 0:3].clone()
        lin = root_states[1, 7:10].clone()
        ang = root_states[1, 10:13].clone()
        return x, lin, ang

    def set_pose(self, root_states: torch.Tensor, pos_xyz, *, zero_velocity: bool = True):
        """
        直接写入球的位姿（位置+清零速度可选），并 set_actor_root_state_tensor 生效。
        """
        x, y, z = float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
        root_states[1, 0] = x
        root_states[1, 1] = y
        root_states[1, 2] = z
        if zero_velocity:
            root_states[1, 7:13] = 0.0
        self.controller.gym.set_actor_root_state_tensor(self.controller.sim, gymtorch.unwrap_tensor(root_states))

    # ------ 环带重置 / 远处重刷 ------
    def reset_ring(self, root_states: torch.Tensor,
                   r_min: float, r_max: float,
                   base_xy=None, z: float = None,
                   theta_range: tuple = (-np.pi, np.pi)):
        """
        把球刷在“以机器人为圆心”的环形带内：r∈[r_min, r_max]，角度均匀
        与你原来在 ChaseBallEnv._reset_ball_positions 里的逻辑等价:contentReference[oaicite:0]{index=0}。
        """
        if base_xy is None:
            base_x = float(root_states[0, 0].item())
            base_y = float(root_states[0, 1].item())
        else:
            base_x, base_y = float(base_xy[0]), float(base_xy[1])

        z = self.default_z if z is None else float(z)

        theta = np.random.uniform(theta_range[0], theta_range[1])
        r = np.random.uniform(float(r_min), float(r_max))
        x = base_x + r * np.cos(theta)
        y = base_y + r * np.sin(theta)

        self.set_pose(root_states, (x, y, z), zero_velocity=True)

    def reset_at_feet(self,
                      root_states: torch.Tensor,
                      base_pos: torch.Tensor,
                      base_quat: torch.Tensor,
                      forward_dist: float = 0.18,
                      lateral_offset: float = 0.0,
                      z: float = None,
                      zero_velocity: bool = True):
        """
        将球重置到机器人“脚下正前方”，使机器人面向球。
        - root_states: Isaac Gym 的 actor_root_state_tensor (wrapped)，shape [num_actors, 13]
        - base_pos: 机器人底座位置张量，shape [1, 3]（与 PassBallEnv 的 self.base_pos 一致）
        - base_quat: 机器人底座朝向四元数，shape [1, 4]（与 PassBallEnv 的 self.base_quat 一致）
        - forward_dist: 球心相对机器人基座的前向距离（米）。建议略大于球半径（0.11）再加 5~7cm 缓冲
        - lateral_offset: 侧向微偏移（米）；默认 0（正前）
        - z: 球心高度；默认用 self.default_z（球半径 + 微小地面抬升）
        - zero_velocity: 是否将球的线/角速度清零
        """
        # 目标高度
        z = self.default_z if z is None else float(z)

        # 世界系前向向量：把机体系 [1,0,0] 旋到世界系
        device = base_pos.device
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3)
        fwd_world = quat_rotate(base_quat[0:1], forward_local).squeeze(0)  # (3,)

        # 仅用平面分量，并归一化；若几乎为零，回退到 X 正方向
        fwd_xy = fwd_world[:2]
        norm = torch.norm(fwd_xy)
        if float(norm) < 1e-6:
            fwd_xy = torch.tensor([1.0, 0.0], device=device)
        else:
            fwd_xy = fwd_xy / norm

        # 可选：侧向单位向量（逆时针旋转 90°）
        perp_xy = torch.stack(torch.unbind(torch.tensor([-fwd_xy[1], fwd_xy[0]], device=device)))

        # 目标平面位置
        base_xy = base_pos[0, :2]  # (2,)
        target_xy = base_xy + forward_dist * fwd_xy + lateral_offset * perp_xy

        # 写入 root_states 中球的位姿（球索引为 1：紧随机器人之后）
        x = float(target_xy[0].item())
        y = float(target_xy[1].item())
        self.set_pose(root_states, (x, y, z), zero_velocity=zero_velocity)

        return (x, y, z)
        
    def respawn_far(self, root_states: torch.Tensor,
                    r_min: float = 2.0, r_max: float = 8.0,
                    base_xy=None, z: float = None):
        """
        多目标连击：不结束回合，直接把球重刷到远处。
        对应你原来的 respawn_ball_far 实现:contentReference[oaicite:1]{index=1}。
        """
        if base_xy is None:
            base_x = float(root_states[0, 0].item())
            base_y = float(root_states[0, 1].item())
        else:
            base_x, base_y = float(base_xy[0]), float(base_xy[1])

        z = self.default_z if z is None else float(z)

        theta = np.random.uniform(-np.pi, np.pi)
        r = np.random.uniform(float(r_min), float(r_max))
        x = base_x + r * np.cos(theta)
        y = base_y + r * np.sin(theta)

        self.set_pose(root_states, (x, y, z), zero_velocity=True)

    # ------ 调试 ------
    def debug_print(self, body_states: torch.Tensor, base_pos: torch.Tensor, env_id: int = 0):
        num_robot_bodies = self.controller.num_bodies_robot
        print(f"Robot body positions (env {env_id}):")
        for i in range(num_robot_bodies):
            pos = body_states[env_id, i, 0:3]
            print(f"  Body {i}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        ball_pos = body_states[env_id, self.ball_idx, 0:3]
        print(f"Ball position (env {env_id}): x={ball_pos[0]:.3f}, y={ball_pos[1]:.3f}, z={ball_pos[2]:.3f}")
        base = base_pos[env_id, :3]
        print(f"Robot base position (env {env_id}): x={base[0]:.3f}, y={base[1]:.3f}, z={base[2]:.3f}")
