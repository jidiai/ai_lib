from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "critic_1000.pth")

STATE_DIM = 2
ACTION_DIM = 3
HIDDEN_SIZE = 100
NUM_HIDDEN_LAYER = 1


class Dueling_Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear_hid = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear_hid(x1))
        x2 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear_hid(x2))

        # value
        y1 = self.linear2(x1)
        # advantage
        y2 = self.linear3(x2)
        x3 = y1 + y2 - y2.mean(dim=1, keepdim=True)

        return x3

value_fn = Dueling_Critic(input_size=STATE_DIM,
                          output_size=ACTION_DIM,
                          hidden_size=HIDDEN_SIZE)
value_fn.load_state_dict(torch.load(model_path, map_location='cpu'))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).unsqueeze(0)
    action = torch.argmax(value_fn(obs)).item()
    onehot_a = [0,0,0]
    onehot_a[action] = 1
    return [onehot_a]