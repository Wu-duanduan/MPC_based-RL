import numpy as np
import scipy.signal
import torch
import torch.nn as nn


def combined_shape(length, shape=None):  # 返回一个元组(x,y)
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)  # ()可以理解为元组构造函数，*号将shape多余维度去除


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.pi(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.sr = MLPSRFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.s = MLPSFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.r = MLPRFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()


class MLPSRFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.sr = mlp([obs_dim + act_dim] + list(hidden_sizes) + [obs_dim + 1], activation)

    def forward(self, obs, act):
        sr = self.sr(torch.cat([obs, act], dim=-1))
        return sr


class MLPSFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.s = mlp([obs_dim + act_dim] + list(hidden_sizes) + [obs_dim], activation)

    def forward(self, obs, act):
        s = self.s(torch.cat([obs, act], dim=-1))
        return s


class MLPRFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.r = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        r = self.r(torch.cat([obs, act], dim=-1))
        return torch.squeeze(r, -1)  # Critical to ensure q has right shape.
