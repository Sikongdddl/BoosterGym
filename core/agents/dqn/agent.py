# core/agents/dqn/agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .network import QNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim=6, device='cpu',
                 lr=3e-4, gamma=0.99, batch_size=128,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=30000,
                 tau=0.005, dueling=True, buffer_capacity=100000):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.q_net = QNetwork(state_dim, action_dim, dueling=dueling).to(device)
        self.target_net = QNetwork(state_dim, action_dim, dueling=dueling).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # epsilon 线性退火（按高层决策步）
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.frame = 0   # 高层决策步计数（在 select_action 里自增）
        self.epsilon = eps_start

        self.step_count = 0  # 真正的更新次数
        self.last_loss = None

    def _update_epsilon(self):
        self.frame += 1
        frac = min(1.0, self.frame / float(self.eps_decay_steps))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state):
        # state: np.ndarray shape (state_dim,)
        self._update_epsilon()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def push(self, *args):
        self.replay_buffer.push(*args)

    def _soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return False, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # 类型统一，防止 object 数组
        states      = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions, dtype=torch.int64,   device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)

        # 当前 Q
        q_values = self.q_net(states).gather(1, actions)

        # ---- Double DQN target ----
        with torch.no_grad():
            next_q_online = self.q_net(next_states)                      # 在线网选动作
            next_actions  = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions)  # 目标网估值
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        # Huber loss 更稳
        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()

        # 软更新目标网
        self._soft_update()

        self.step_count += 1
        self.last_loss = float(loss.item())
        return True, self.last_loss
