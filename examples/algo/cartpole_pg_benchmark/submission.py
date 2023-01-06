from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "policy_1000.pth")

STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_SIZE = 64
NUM_HIDDEN_LAYER = 1


class Actor(nn.Module):
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

        # self.affine1 = nn.Linear(self.input_size, 128)
        # self.affine2 = nn.Linear(128, self.output_size)

    def forward(self, x):
        # x = F.relu(self.affine1(x))
        # action_scores = self.affine2(x)
        x = F.relu(self.linear_in(x))
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        action_scores = self.linear_out(x)

        return F.softmax(action_scores, dim=1)


policy = Actor(input_size=STATE_DIM,
               output_size=ACTION_DIM,
               hidden_size=HIDDEN_SIZE,
               num_hidden_layer=NUM_HIDDEN_LAYER)
policy.load_state_dict(torch.load(model_path))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).unsqueeze(0)
    probs = policy(obs)
    action = torch.argmax(probs)
    onehot_a = [0,0]
    onehot_a[action.item()] = 1
    return [onehot_a]

