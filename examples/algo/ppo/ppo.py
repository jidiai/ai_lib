import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from algo.ppo.Network import Actor, Critic

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action", "a_logit"]


class PPO(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.update_freq = args.update_freq
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.gamma = args.gamma
        self.hidden_size = args.hidden_size

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size)
        self.critic = Critic(self.state_dim, 1, self.hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_net_optimizer = optim.Adam(self.critic.parameters(), lr=self.c_lr)

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        self.counter = 0
        self.training_step = 0

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train=True):
        state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        logits = self.actor(state).detach()
        action = Categorical(torch.Tensor(logits)).sample()
        return {"action": action.item(),
                "a_logit": logits[:, action.item()].item()}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):

        data_length = len(self.memory.item_buffers["rewards"].data)
        data = self.memory.get_trajectory()

        transitions = {
            "o_0": np.array(data['states']),
            "r_0": data['rewards'],
            "u_0": np.array(data['action']),
            "log_prob": np.array(data['a_logit'])
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float)
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(-1, 1)
        reward = transitions["r_0"]
        old_action_log_prob = torch.tensor(transitions["log_prob"], dtype=torch.float).view(-1, 1)

        # 计算reward-to-go
        R = 0
        Gt = []
        for r in reward[::-1]: # 反过来
            R = r[0] + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        for i in range(self.update_freq):
            for index in BatchSampler(SubsetRandomSampler(range(data_length)), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic(obs[index])
                delta = Gt_index - V
                advantage = delta.detach()

                action_prob = self.actor(obs[index]).gather(1, action[index])

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
        self.memory.item_buffer_clear()

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)

    def load(self, actor_net, critic_net):
        self.actor.load_state_dict(torch.load(actor_net))
        self.critic.load_state_dict(torch.load(critic_net))
