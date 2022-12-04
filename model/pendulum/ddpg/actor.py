from utils.episode import EpisodeKey

import torch
import torch.nn as nn
import numpy as np


from torch.autograd import Variable


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
        out_dim = 1
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
        if EpisodeKey.CUR_OBS in kwargs:
            observations = to_tensor(kwargs[EpisodeKey.CUR_OBS])
            # action_masks = to_tensor(kwargs[EpisodeKey.ACTION_MASK])
        elif EpisodeKey.NEXT_OBS in kwargs:
            observations = to_tensor(kwargs[EpisodeKey.NEXT_OBS])
            # action_masks = to_tensor(kwargs[EpisodeKey.NEXT_ACTION_MASK])
        else:
            raise NotImplementedError
        # print('actor deviec =', self.actor.device)
        # obs = torch.tensor(kwargs[EpisodeKey.CUR_OBS]).float()
        pi = self.actor(observations)
        return pi
