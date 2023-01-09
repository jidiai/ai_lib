from pathlib import Path
import os
current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "critic_1000.pth")

STATE_DIM = 6
ACTION_DIM = 3
HIDDEN_SIZE = 64
NUM_HIDDEN_LAYER = 1

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        return x

value_fn = Critic(input_size=STATE_DIM,
                  output_size=ACTION_DIM,
                  hidden_size=HIDDEN_SIZE,
                  num_hidden_layer=NUM_HIDDEN_LAYER)
value_fn.load_state_dict(torch.load(model_path))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).unsqueeze(0)
    action = torch.argmax(value_fn(obs))
    onehot_a = [0,0,0]
    onehot_a[action.item()] = 1
    return [onehot_a]