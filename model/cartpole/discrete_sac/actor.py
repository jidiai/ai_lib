from utils.episode import EpisodeKey

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal


def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        arr = torch.FloatTensor(arr)
    return arr


class Actor(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        in_dim = observation_space.shape[0]
        out_dim = action_space.n
        hidden_size = 64

        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_dim)
        )

        self._init()

    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.uniform_(module.bias, 0, 0.001)
            else:
                pass

    def forward(self, **kwargs):

        # print('actor deviec =', self.actor.device)
        obs = torch.tensor(kwargs[EpisodeKey.CUR_OBS]).float()
        logits = self.actor(obs)
        return logits
