import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class DQN(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_size = 100

        self.critic_eval = Critic(self.state_dim, self.hidden_size, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.hidden_size, self.action_dim)

        self.buffer = []

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))

action_dim = 2
state_dim = 4
agent = DQN(state_dim, action_dim)
actor_net = os.path.dirname(os.path.abspath(__file__)) + '/critic_net.pth'
agent.load(actor_net)

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
    action = agent.choose_action(np.array(obs_list[0]))
    action_ = action_wrapper([action])
    return action_