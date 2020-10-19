import torch
import torch.nn as nn
from gym.spaces import Discrete


def create_model(env):
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        return CategoricalAgent(obs_dim, env.action_space.n)
    else:
        return ContinuousAgent(obs_dim, env.action_space.shape[0])


class CategoricalAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, action_dim))

    @torch.no_grad()
    def forward(self, x):
        logits = self.model(x)
        return torch.argmax(logits).item()


class ContinuousAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, action_dim), nn.Tanh())

    @torch.no_grad()
    def forward(self, x):
        return self.model(x).squeeze().numpy()
