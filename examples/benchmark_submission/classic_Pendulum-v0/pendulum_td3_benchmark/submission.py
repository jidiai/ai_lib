from pathlib import Path
import os
import numpy as np

current_path = Path(__file__).resolve().parent
actor_path = os.path.join(current_path, "actor1000.pth")
print(actor_path)

STATE_DIM = 3
ACTION_DIM = 1
HIDDEN_SIZE = 100
NUM_HIDDEN_LAYER = 1

import torch
import torch.nn as nn


class ContinuousTanhActor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_size, action_loc, action_scale,
                 num_hidden_layer=1):
        super(ContinuousTanhActor, self).__init__()

        self.input_size = state_dim
        self.hidden_size = hidden_size
        self.output_size = action_dim

        actor_linear_in = nn.Linear(self.input_size, self.hidden_size)
        actor_linear_out = nn.Linear(self.hidden_size, action_dim)
        actor_hidden_list = []
        if num_hidden_layer > 0:
            for _ in range(num_hidden_layer):
                actor_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                actor_hidden_list.append(nn.ReLU())
        self.net = nn.Sequential(
            actor_linear_in,
            nn.ReLU(),
            *actor_hidden_list,
            actor_linear_out,
            nn.Tanh()
        )
        self.action_loc = action_loc
        self.action_scale = action_scale
    def forward(self, x):
        x = self.net(x)
        return x*self.action_scale+self.action_loc

policy = ContinuousTanhActor(state_dim=STATE_DIM,
                             action_dim=ACTION_DIM,
                             hidden_size=HIDDEN_SIZE,
                             num_hidden_layer=NUM_HIDDEN_LAYER,
                             action_loc=0,action_scale=2)
policy.load_state_dict(torch.load(actor_path))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).view(1,-1)
    action = policy(obs).item()

    return [np.array([action])]