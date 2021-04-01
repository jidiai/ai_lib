import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from torch.distributions import Categorical
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 32)

        self.action_head = nn.Linear(32, 2)
        self.value_head = nn.Linear(32, 1) # Scalar Value

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value

def select_action(state):
    state = torch.from_numpy(np.array(state)).float()
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

policy = Policy()
policy_net = os.path.dirname(os.path.abspath(__file__)) + '/policy_net.pth'
policy.load_state_dict(torch.load(policy_net))

def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    action_a = joint_action[0]
    each = [0] * 2
    each[action_a] = 1
    action_one_hot = [[each]]
    joint_action_.append([action_one_hot[0][0]])
    return joint_action_

def my_controller(obs_list, action_space_list, obs_space_list):
    action = select_action(obs_list[0])
    action_ = action_wrapper([action])
    return action_