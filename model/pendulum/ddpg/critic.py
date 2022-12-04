import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
from gym.spaces import Box, Discrete
import numpy as np
from utils.episode import EpisodeKey

import torch
import torch.nn as nn


def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        arr = torch.FloatTensor(arr)
    return arr


class Critic(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space: Box,
        action_space: Discrete,
        custom_config,
        initialization,
    ):
        super().__init__()
        # game = pyspiel.load_game('kuhn_poker')
        # policy=policy_lib.TabularPolicy(game)
        # assert action_space.n==policy.action_probability_array.shape[-1]
        # self.q_table=nn.Parameter(torch.zeros(size=policy.action_probability_array.shape))
        # torch.nn.init.uniform_(self.q_table,a=0,b=0.001)
        in_dim = observation_space.shape[0] + action_space.shape[0]
        out_dim = 1
        hidden_size = 64

        self.q_net = nn.Sequential(
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

        obs_action = kwargs[EpisodeKey.OBS_ACTION]
        q_values = self.q_net(obs_action)
        # mask out invalid actions
        # q_values = action_masks * q_values + (1 - action_masks) * (-10e9)
        return q_values
