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
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        game = pyspiel.load_game("kuhn_poker")
        policy = policy_lib.TabularPolicy(game)
        assert action_space.n == policy.action_probability_array.shape[-1]
        self.q_table = nn.Parameter(
            torch.zeros(size=policy.action_probability_array.shape)
        )
        torch.nn.init.uniform_(self.q_table, a=0, b=0.001)

    def forward(self, **kwargs):
        observations = to_tensor(kwargs[EpisodeKey.CUR_OBS])
        action_masks = to_tensor(kwargs[EpisodeKey.ACTION_MASK])

        # observations are encoded as index to q_table
        assert len(observations.shape) == 2
        observations = observations.reshape(-1)
        observations = observations.long()
        q_values = self.q_table[observations]
        # mask out invalid actions
        q_values = action_masks * q_values + (1 - action_masks) * (-1e9)
        return q_values
