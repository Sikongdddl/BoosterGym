# envs/components/ball_world.py
import numpy as np
import torch
from isaacgym import gymtorch

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
