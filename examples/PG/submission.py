import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from torch.distributions import Categorical
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def select_action(state):
    state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


env_type = "classic_CartPole-v0"
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