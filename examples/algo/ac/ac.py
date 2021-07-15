import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
print(base_dir)
from env.chooseenv import make

#Parameters
env = make('classic_CartPole-v0')
action_space = env.action_dim
observation_space = env.input_dimension.shape[0]
print(action_space, observation_space)

#Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 20000
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(observation_space, 32)

        self.action_head = nn.Linear(32, action_space)
        self.value_head = nn.Linear(32, 1) # Scalar Value

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    action_a = joint_action
    each = [0] * env.action_dim
    each[action_a] = 1
    action_one_hot = [[each]]
    joint_action_.append([action_one_hot[0][0]])
    return joint_action_

def finish_episode():
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r[0] + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob , value), r in zip(save_actions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(np.array(state))
            action = action_wrapper(action)
            state, reward, done, _, _ = env.step(action)
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        torch.save(model.state_dict(), 'policy_net.pth')


if __name__ == '__main__':
    main()