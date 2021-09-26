import random
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np

from networks.critic import Critic

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action"]


class DDQN(object):
    def __init__(self, args):
        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.hidden_size = args.hidden_size
        self.lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optimizer.Adam(self.critic_eval.parameters(), lr=self.lr)

        # exploration
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)

        # 更新target网
        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train):
        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                action = random.randrange(self.action_dim)

            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                action = torch.argmax(self.critic_eval(observation)).item()
        else:
            observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.critic_eval(observation)).item()

        return {"action": action}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):

        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return

        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data['states']),
            "o_next_0": np.array(data['states_next']),
            "r_0": np.array(data['rewards']).reshape(-1, 1),
            "u_0": np.array(data['action']),
            "d_0": np.array(data['dones']).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float)
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(transitions["r_0"], dtype=torch.float)
        done = torch.tensor(transitions["d_0"], dtype=torch.float)

        q_eval = self.critic_eval(obs).gather(1, action)
        q_eval_next = self.critic_eval(obs_)
        max_action = torch.argmax(q_eval_next, dim=1).unsqueeze(1)
        q_next = self.critic_target(obs_).gather(1, max_action).detach()
        q_target = reward + self.gamma * q_next * (1 - done)

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        return loss

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_eval.state_dict(), model_critic_path)

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))


