import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from networks.actor_critic import ActorCritic

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer

eps = np.finfo(np.float32).eps.item()


def get_trajectory_property():
    return ["action", 'log_prob', 'value']


class AC(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.hidden_size = args.hidden_size

        self.lr = args.c_lr
        self.gamma = args.gamma

        if args.given_net:
            self.policy = args.network(self.state_dim, self.action_dim, self.hidden_size)
        else:
            self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.save_actions = []
        self.save_value = []
        self.rewards = []

        self.buffer_size = args.buffer_capacity
        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def inference(self, observation, train=True):
        if train:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            probs, value = self.policy(state)
            m = Categorical(probs)
            action = m.sample()
        else:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            probs, value = self.policy(state)
            m = Categorical(probs)
            action = torch.argmax(probs)
        return {
            "action": action.item(),
            "log_prob": m.log_prob(action),
            "value": value
        }

    def learn(self):
        self.rewards = self.memory.item_buffers["rewards"].data
        self.save_actions = self.memory.item_buffers["log_prob"].data
        self.save_value = self.memory.item_buffers["value"].data

        R = 0
        policy_loss = []
        value_loss = []
        rewards = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for (log_prob, value, r) in zip(self.save_actions, self.save_value, rewards):
            reward = r - value.item()

            policy_loss.append(-log_prob * reward)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.save_actions[:]
        del self.save_value[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "policy_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_critic_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))