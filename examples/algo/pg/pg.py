import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from networks.actor import Actor

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer

eps = np.finfo(np.float32).eps.item()


def get_trajectory_property():
    return ["action"]


class PG(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.lr = args.c_lr
        self.gamma = args.gamma

        self.policy = Actor(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.saved_log_probs = []
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
            probs = self.policy(state)
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
        else:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            probs = self.policy(state)
            action = torch.argmax(probs)
        return {"action": action.item()}

    def learn(self):
        self.rewards = self.memory.item_buffers["rewards"].data
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r[0] + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "policy_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_critic_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))
