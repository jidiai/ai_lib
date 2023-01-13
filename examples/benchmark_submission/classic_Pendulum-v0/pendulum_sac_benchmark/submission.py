from pathlib import Path
import os
current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "actor_1000.pth")

STATE_DIM = 3
ACTION_DIM = 1
HIDDEN_SIZE = 64
NUM_HIDDEN_LAYER = 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        tanh=False,
        action_high=2,
        action_low=-2,
    ):
        super(GaussianActor, self).__init__()

        self.linear_in = nn.Linear(state_dim, hidden_dim)
        self.linear_hid = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.logstd_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        self.tanh = tanh
        if tanh:  # normalise the action
            self.action_scale = torch.FloatTensor([(action_high - action_low) / 2.0])
            self.action_bias = torch.FloatTensor([(action_high + action_low) / 2.0])

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        x = F.relu(self.linear_hid(x))
        mean = self.mean_linear(x)
        log_std = self.logstd_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, logstd = self.forward(state)
        std = logstd.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        if self.tanh:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = mean

        return action, log_prob, mean

policy = GaussianActor(state_dim=STATE_DIM,
                       hidden_dim=HIDDEN_SIZE,
                       action_dim=ACTION_DIM, tanh=False)
policy.load_state_dict(torch.load(model_path))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).view(1,-1)
    _,_,action = policy.sample(obs)
    return [np.array([action.item()])]

