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

from algo.ddpg.Network import Actor, Critic, ContinuousActor

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
        self.use_cuda = args.use_cuda
        self.policy_type = args.policy_type
        self.continuous_action_min = args.continuous_action_min
        self.continuous_action_max = args.continuous_action_max

        self.update_freq = args.update_freq

        if self.policy_type == 'discrete':
            self.actor = Actor(
                self.state_dim, self.action_dim, self.hidden_size, args.num_hidden_layer
            )
            self.actor_target = Actor(
                self.state_dim, self.action_dim, self.hidden_size, args.num_hidden_layer
            )
        elif self.policy_type == 'continuous':
            self.actor = ContinuousActor(
                self.state_dim, self.action_dim, self.hidden_size, args.num_hidden_layer
            )
            self.actor_target = ContinuousActor(
                self.state_dim, self.action_dim, self.hidden_size, args.num_hidden_layer
            )
        else:
            raise NotImplementedError

        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            self.state_dim, self.action_dim, self.hidden_size, args.num_hidden_layer
        )
        self.critic_target = Critic(
            self.state_dim, self.action_dim, self.hidden_size, args.num_hidden_layer
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.to_cuda()

        self.actor_optim = optimizer.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optimizer.Adam(self.critic.parameters(), lr=self.critic_lr)

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        self.eps = args.epsilon
        self.epsilon_end = args.epsilon_end

    def to_cuda(self):
        if self.use_cuda:
            self.actor.to('cuda')
            self.actor_target.to('cuda')
            self.critic.to('cuda')
            self.critic_target.to('cuda')

    def tensor_to_cuda(self, tensor):
        if self.use_cuda:
            return tensor.to('cuda')
        else:
            return tensor

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train=True):

        self.eps *= 0.99999
        self.eps = max(self.eps, self.epsilon_end)

        if self.policy_type == 'discrete':
            if train:
                if random.random() > self.eps:
                    state = self.tensor_to_cuda(torch.tensor(
                        observation, dtype=torch.float).unsqueeze(0))
                    logits = self.actor(state).detach().cpu().numpy()
                else:
                    logits = np.random.uniform(low=0, high=1, size=(1, self.action_dim))
                action = Categorical(torch.Tensor(logits)).sample()
            else:
                state = self.tensor_to_cuda(torch.tensor(
                    observation, dtype=torch.float).unsqueeze(0))
                logits = self.actor(state).detach().cpu().numpy()
                action = Categorical(torch.Tensor(logits)).sample()
            return {"action": action, "logits": logits}
        elif self.policy_type=='continuous':
            state = self.tensor_to_cuda(torch.tensor(
                observation, dtype=torch.float).unsqueeze(0))
            logits = self.actor(state).detach().cpu().numpy()

            # mid = (self.continuous_action_max+self.continuous_action_min)/2
            # scale = self.continuous_action_max-mid
            # tanh_a = torch.tanh(logits.squeeze(0))
            # action = tanh_a*scale + mid
            # action = tanh_a*scale + mid
            # if train:
            #     if random.random() > self.eps:
            #         logits += np.random.normal(0, 0.2)

            # action = np.clip(logits.squeeze(0), self.continuous_action_min, self.continuous_action_max)
            action = logits.squeeze(0)

            return {"action": action, "logits": logits}

    # if train:
        # self.eps *= 0.99999
        # self.eps = max(self.eps, self.epsilon_end)
        # state = self.tensor_to_cuda(torch.tensor(
        #     observation, dtype=torch.float).unsqueeze(0))
        # logits = self.actor(state).detach()
        #
        # if self.policy_type == 'discrete':
        #     if random.random() > self.eps:
        #         pass
        #     else:
        #         logits = torch.Tensor(np.random.uniform(low=0, high=1, size=(1, 2)))
        #
        #     action = Categorical(logits=logits).sample()
        #     return {"action": action.cpu(), "logits": logits.squeeze(0).cpu().numpy()}
        # elif self.policy_type == 'continuous':
        #     mid = (self.continuous_action_max+self.continuous_action_min)/2
        #     scale = self.continuous_action_max-mid
        #     tanh_a = torch.tanh(logits)
        #     action = tanh_a*scale + mid
        #     return {"action": action.squeeze(0).cpu().numpy(), "logits": logits.squeeze(0).cpu().numpy()}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):
        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return {}

        training_results = {
            "policy_loss": [],
            "value_loss": [],
        }

        for _ in range(self.update_freq):

            data = self.memory.sample(self.batch_size)

            transitions = {
                "o_0": np.array(data["states"]),
                "o_next_0": np.array(data["states_next"]),
                "r_0": np.array(data["rewards"]).reshape(-1, 1),
                "u_0": np.array(data["logits"]),
                "d_0": np.array(data["dones"]).reshape(-1, 1),
            }

            obs = self.tensor_to_cuda(torch.tensor(transitions["o_0"], dtype=torch.float))
            obs_ = self.tensor_to_cuda(torch.tensor(transitions["o_next_0"], dtype=torch.float))
            if self.policy_type == 'continuous':
                action = self.tensor_to_cuda(torch.tensor(transitions["u_0"], dtype=torch.float).squeeze(-1))
            elif self.policy_type == 'discrete':
                action = self.tensor_to_cuda(torch.tensor(transitions["u_0"], dtype=torch.float).squeeze())
            reward = self.tensor_to_cuda(torch.tensor(transitions["r_0"], dtype=torch.float).view(
                self.batch_size, -1
            ))
            done = self.tensor_to_cuda(
                torch.tensor(transitions["d_0"], dtype=torch.float)
                .squeeze()
                .view(self.batch_size, -1)
            )

            with torch.no_grad():
                a1 = self.actor_target(obs_)
                # if self.policy_type == 'continuous':
                    # mid = (self.continuous_action_max + self.continuous_action_min) / 2
                    # scale = self.continuous_action_max - mid
                    # tanh_a = torch.tanh(a1)
                    # a1 = tanh_a * scale + mid
                    # a1 = torch.clip(a1, self.continuous_action_min, self.continuous_action_max)
                y_true = reward + self.gamma * (1 - done) * self.critic_target(obs_, a1)
            y_pred = self.critic(obs, action)

            loss_fn = nn.MSELoss()
            value_loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)

            grad_dict = {}
            # for name, param in self.critic.named_parameters():
            #     grad_dict[f'Critic/{name} gradient']=param.grad.mean().item()

            self.critic_optim.step()

            cur_a = self.actor(obs)
            # if self.policy_type == 'continuous':
                # mid = (self.continuous_action_max + self.continuous_action_min) / 2
                # scale = self.continuous_action_max - mid
                # tanh_a = torch.tanh(cur_a)
                # cur_a = tanh_a * scale + mid
                # cur_a = torch.clip(cur_a, self.continuous_action_min, self.continuous_action_max)

            actor_loss = -torch.mean(self.critic(obs, cur_a))
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)

            # for name, param in self.actor.named_parameters():
            #     grad_dict[f'Actor/{name} gradient']=param.grad().mean().item()

            self.actor_optim.step()

            self.soft_update(self.critic_target, self.critic, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)

            training_results['policy_loss'].append(actor_loss.detach().cpu().numpy())
            training_results['value_loss'].append(value_loss.detach().cpu().numpy())

        for tag,v in training_results.items():
            training_results[tag] = np.mean(v)
        training_results.update(grad_dict)

        return training_results

    def soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

    def load(self, load_path, i):
        self.actor.load_state_dict(
            torch.load(str(load_path) + "/actor_net_{}.pth".format(i))
        )

    def scale_noise(self, decay):
        self.actor.noise.scale = self.actor.noise.scale * decay
