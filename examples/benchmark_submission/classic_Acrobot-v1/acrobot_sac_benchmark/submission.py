from pathlib import Path
import os
current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "actor_1000.pth")

STATE_DIM = 6
ACTION_DIM = 3
HIDDEN_SIZE = 64
NUM_HIDDEN_LAYER = 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
class CategoricalActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CategoricalActor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(
            hidden_dim, action_dim
        )  # should be followed by a softmax layer
        self.apply(weights_init_)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        prob = F.softmax(x, -1)  # [batch_size, action_dim]
        return prob

    def sample(self, state):
        prob = self.forward(state)

        distribution = Categorical(probs=prob)
        sample_action = distribution.sample().unsqueeze(-1)  # [batch, 1]
        z = (prob == 0.0).float() * 1e-8
        logprob = torch.log(prob + z)
        greedy = torch.argmax(prob, dim=-1).unsqueeze(-1)  # 1d tensor

        return sample_action, prob, logprob, greedy

policy=CategoricalActor(state_dim=STATE_DIM,
                        hidden_dim=HIDDEN_SIZE,
                        action_dim=ACTION_DIM)
policy.load_state_dict(torch.load(model_path))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).unsqueeze(0)
    _,_,_,action = policy.sample(obs)
    onehot_a = [0]*ACTION_DIM
    onehot_a[action.item()]=1
    return [onehot_a]


