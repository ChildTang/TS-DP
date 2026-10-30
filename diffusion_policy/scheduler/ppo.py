"""PPO scheduler network (tianshou 0.5).

Module attribute names (``image_encoder``, ``fusion_net``, ``mlp``) are part of
the released PPO checkpoints; do not rename them.
"""
from typing import Sequence

import numpy as np
import torch
from torch import nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic


def _mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers = []
    for hidden_dim in hidden_sizes:
        layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        input_dim = hidden_dim
    return nn.Sequential(*layers)


def _to_tensor(x, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class ImageStateNet(nn.Module):
    """Shared CNN over each camera image, concatenated with the state vector."""

    def __init__(self, observation_space, image_keys: Sequence[str],
            hidden_sizes: Sequence[int], device):
        super().__init__()
        self.device = device
        self.image_keys = list(image_keys)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            image_shape = observation_space[self.image_keys[0]].shape
            image_dim = self.image_encoder(torch.zeros(1, *image_shape)).shape[1]
        state_dim = observation_space['state'].shape[0]
        self.fusion_net = _mlp(image_dim * len(self.image_keys) + state_dim, hidden_sizes)
        self.output_dim = hidden_sizes[-1]

    def forward(self, obs, state=None, info={}):
        features = [self.image_encoder(_to_tensor(obs[k], self.device)) for k in self.image_keys]
        features.append(_to_tensor(obs['state'], self.device))
        return self.fusion_net(torch.cat(features, dim=-1)), state


class VectorNet(nn.Module):
    def __init__(self, observation_space, hidden_sizes: Sequence[int], device):
        super().__init__()
        self.device = device
        self.mlp = _mlp(observation_space.shape[0], hidden_sizes)
        self.output_dim = hidden_sizes[-1]

    def forward(self, obs, state=None, info={}):
        return self.mlp(_to_tensor(obs, self.device)), state


def build_ppo_policy(task, observation_space, action_space,
        hidden_sizes: Sequence[int] = (256, 256), device: str = 'cpu') -> PPOPolicy:
    """Actor-critic with the layout of the released PPO checkpoints."""
    if task.obs_type == 'robomimic':
        net = ImageStateNet(observation_space, task.image_keys, hidden_sizes, device)
    elif task.obs_type == 'pusht':
        net = ImageStateNet(observation_space, ['image'], hidden_sizes, device)
    else:
        net = VectorNet(observation_space, hidden_sizes, device)
    net = net.to(device)
    actor = Actor(net, action_space.n, hidden_sizes=(), device=device).to(device)
    critic = Critic(net, hidden_sizes=(), device=device).to(device)
    # tianshou requires an optimizer; it is never stepped at inference
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters())
    return PPOPolicy(actor, critic, optim,
        dist_fn=torch.distributions.Categorical,
        observation_space=observation_space,
        action_space=action_space,
        action_scaling=False)
