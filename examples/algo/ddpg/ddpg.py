# -*- coding:utf-8  -*-
# Time  : 2021/03/03 16:52
# Author: Yutong Wu

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torch.distributions import Categorical

from algo.ddpg.Network import Actor, Critic

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action", "logits"]


class DDPG(object):
    def __init__(self, args):
        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.hidden_size = args.hidden_size
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau

        self.update_freq = args.update_freq

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optimizer.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optimizer.Adam(self.critic.parameters(), lr=self.critic_lr)

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        self.eps = args.epsilon
        self.epsilon_end = args.epsilon_end

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train=True):
        # if train:
        self.eps *= 0.99999
        self.eps = max(self.eps, self.epsilon_end)
        if random.random() > self.eps:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            logits = self.actor(state).detach().numpy()
        else:
            logits = np.random.uniform(low=0, high=1, size=(1,2))
        action = Categorical(torch.Tensor(logits)).sample()
        return {"action": action,
                "logits": logits}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):
        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return

        for _ in range(self.update_freq):

            data = self.memory.sample(self.batch_size)

            transitions = {
                "o_0": np.array(data['states']),
                "o_next_0": np.array(data['states_next']),
                "r_0": np.array(data['rewards']).reshape(-1, 1),
                "u_0": np.array(data['logits']),
                "d_0": np.array(data['dones']).reshape(-1, 1),
            }

            obs = torch.tensor(transitions["o_0"], dtype=torch.float)
            obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float)
            action = torch.tensor(transitions["u_0"], dtype=torch.float).squeeze()
            reward = torch.tensor(transitions["r_0"], dtype=torch.float).view(self.batch_size, -1)
            done = torch.tensor(transitions["d_0"], dtype=torch.float).squeeze().view(self.batch_size, -1)

            with torch.no_grad():
                a1 = self.actor_target(obs_)
                y_true = reward + self.gamma * (1 - done) * self.critic_target(obs_, a1)
            y_pred = self.critic(obs, action)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            loss = -torch.mean(self.critic(obs, self.actor(obs)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

            self.soft_update(self.critic_target, self.critic, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)

    def soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_actor_path)

    def load(self, load_path, i):
        self.actor.load_state_dict(torch.load(str(load_path) + '/actor_net_{}.pth'.format(i)))

    def scale_noise(self, decay):
        self.actor.noise.scale = self.actor.noise.scale * decay
