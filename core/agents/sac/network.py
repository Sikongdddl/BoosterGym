import torch
import torch.nn as nn
import torch.nn.functional as F
import math

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def mlp(sizes, act=nn.ReLU, last_act=None):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [act()]
        elif last_act is not None:
            layers += [last_act()]
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim], act=nn.ReLU)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self(x)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        # 计算 log_prob（tanh 修正项）
        log_prob = (-0.5 * ((pre_tanh - mu) / (std + 1e-8))**2 - log_std
                   - 0.5*math.log(2*math.pi)).sum(dim=-1, keepdim=True)
        # tanh 修正：log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x))
        log_prob -= torch.log(1 - a.pow(2) + 1e-8).sum(dim=-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)  # 同时返回确定性动作（eval 用）

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q = mlp([state_dim + action_dim, hidden_dim, hidden_dim, 1], act=nn.ReLU)

    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=-1))
 
