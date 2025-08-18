import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from core.agents.base_agent import BaseAgent
from .network import GaussianPolicy, QNetwork
from .replay_buffer import ReplayBuffer

class SACAgent(BaseAgent):
    def __init__(self,
                 state_dim,
                 action_dim,
                 device='cpu',
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=None,                 # 若为 None 则启用自动温度
                 target_entropy=None,        # 默认为 -action_dim
                 batch_size=256,
                 buffer_capacity=200000,
                 action_low=None, action_high=None):  # 连续动作的物理范围（numpy 形状[action_dim]）
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.pi_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        # 温度
        if alpha is None:
            # 自动温度
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.auto_alpha = True
            self.target_entropy = float(-(action_dim)) if target_entropy is None else float(target_entropy)
        else:
            self.alpha = float(alpha)
            self.auto_alpha = False

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # 动作缩放到物理范围
        if action_low is None: action_low = -np.ones(action_dim, dtype=np.float32)
        if action_high is None: action_high =  np.ones(action_dim, dtype=np.float32)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)

    # 将 tanh(-1,1) 空间动作缩放到物理范围
    def _scale_action(self, a_tanh):
        # a_tanh: torch.Tensor [..., action_dim] in (-1,1)
        return (self.action_high + self.action_low)/2.0 + a_tanh * (self.action_high - self.action_low)/2.0

    def select_action(self, state, eval_mode=False):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_tanh, _, a_mean = self.policy.sample(s)
            a_use = a_mean if eval_mode else a_tanh
            a_scaled = self._scale_action(a_use).squeeze(0)
        return a_scaled.cpu().numpy()

    def push(self, *args):
        self.replay_buffer.push(*args)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return False, None

        s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)
        s   = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a   = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r   = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2  = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d   = torch.as_tensor(d.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---- 计算 α（自动温度）----
        if self.auto_alpha:
            with torch.no_grad():
                a2_tanh, logp2, _ = self.policy.sample(s2)
            alpha = self.log_alpha.exp()
            alpha_loss = (alpha * (-logp2 - self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha = torch.tensor(self.alpha, device=self.device)

        # ---- 目标 Q ----
        with torch.no_grad():
            a2_tanh, logp2, _ = self.policy.sample(s2)
            a2 = self._scale_action(a2_tanh)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            q_t_min = torch.min(q1_t, q2_t) - alpha * logp2
            y = r + (1.0 - d) * self.gamma * q_t_min

        # ---- 更新 Q1/Q2 ----
        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)
        self.q1_optim.zero_grad(); q1_loss.backward(); self.q1_optim.step()
        self.q2_optim.zero_grad(); q2_loss.backward(); self.q2_optim.step()

        # ---- 更新策略（重参数化）----
        a_tanh, logp, _ = self.policy.sample(s)
        a_s = self._scale_action(a_tanh)
        q1_pi = self.q1(s, a_s)
        q2_pi = self.q2(s, a_s)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (alpha * logp - q_pi).mean()
        self.pi_optim.zero_grad(); pi_loss.backward(); self.pi_optim.step()

        # ---- 软更新目标网 ----
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        # 返回一个标量 loss 便于 TB 记录
        total_loss = (q1_loss + q2_loss + pi_loss).item()
        return True, total_loss

    @torch.no_grad()
    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ---- 只保存“权重” ----
    def save(self, path):
        pkg = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
        }
        if hasattr(self, "log_alpha"):
            pkg["log_alpha"] = self.log_alpha.detach().cpu()
        torch.save(pkg, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if "policy" in ckpt: self.policy.load_state_dict(ckpt["policy"])
        if "q1" in ckpt: self.q1.load_state_dict(ckpt["q1"])
        if "q2" in ckpt: self.q2.load_state_dict(ckpt["q2"])
        if "q1_target" in ckpt: self.q1_target.load_state_dict(ckpt["q1_target"])
        if "q2_target" in ckpt: self.q2_target.load_state_dict(ckpt["q2_target"])
        if "log_alpha" in ckpt and hasattr(self, "log_alpha"):
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

