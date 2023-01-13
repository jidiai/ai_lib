import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from examples.algo.ppo.Network import Actor, Critic
from examples.networks.encoder import CNNEncoder

import os
from pathlib import Path
import sys
import random

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from examples.common.buffer import Replay_buffer as buffer


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
        self.use_cuda = args.use_cuda

        if 'cnn' in vars(args):
            self.use_cnn_encoder = True
            cnn_cfg = args.cnn
            # self.cnn_encoder = Flatten()
            self.actor_cnn_encoder = CNNEncoder(input_chanel=cnn_cfg['input_chanel'],
                                          hidden_size=None,
                                          output_size=None,
                                          channel_list=cnn_cfg['channel_list'],
                                          kernel_list=cnn_cfg['kernel_list'],
                                          stride_list=cnn_cfg['stride_list'],
                                          batch_norm=False)
            self.critic_cnn_encoder = CNNEncoder(input_chanel=cnn_cfg['input_chanel'],
                                          hidden_size=None,
                                          output_size=None,
                                          channel_list=cnn_cfg['channel_list'],
                                          kernel_list=cnn_cfg['kernel_list'],
                                          stride_list=cnn_cfg['stride_list'],
                                          batch_norm=False)
            self.actor = Actor(self.hidden_size, self.action_dim, self.hidden_size)
            self.critic = Critic(self.hidden_size, 1, self.hidden_size)
            self.actor_optimizer = optim.Adam(list(self.actor.parameters())+list(self.actor_cnn_encoder.parameters()), lr=self.a_lr)
            self.critic_net_optimizer = optim.Adam(list(self.critic.parameters())+list(self.critic_cnn_encoder.parameters()), lr=self.c_lr)
        else:
            self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size)
            self.critic = Critic(self.state_dim, 1, self.hidden_size)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)
            self.critic_net_optimizer = optim.Adam(self.critic.parameters(), lr=self.c_lr)
        self.to_cuda()

        # pretrained_path = args.pretrain
        # if len(pretrained_path)>0:
        #     encoder_dict = torch.load(pretrained_path['encoder_path'])
        #     self.actor_cnn_encoder.load_state_dict(encoder_dict['actor_cnn_encoder'])
        #     self.critic_cnn_encoder.load_state_dict(encoder_dict['critic_cnn_encoder'])
        #     self.actor.load_state_dict(torch.load(pretrained_path['actor_path']))
        #     self.critic.load_state_dict(torch.load(pretrained_path['critic_path']))


        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)

        self.counter = 0
        self.training_step = 0

    def to_cuda(self):
        if self.use_cuda:
            self.actor.to('cuda')
            self.critic.to('cuda')
            if self.use_cnn_encoder:
                self.actor_cnn_encoder.to('cuda')
                self.critic_cnn_encoder.to('cuda')

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

    def inference(self, observation, train):
        state = self.tensor_to_cuda(torch.tensor(observation, dtype=torch.float).unsqueeze(0))

        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                action = random.randrange(self.action_dim)
            else:
                if self.use_cnn_encoder:
                    state = self.actor_cnn_encoder(state)
                logits = self.actor(state).detach()
                action = Categorical(logits).sample().item()
        else:
            if self.use_cnn_encoder:
                state = self.actor_cnn_encoder(state)
            logits = self.actor(state).detach()
            action = Categorical(logits).sample().item()

        return {"action": action, "a_logit": logits[:, action].item()}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):

        data_length = len(self.memory.item_buffers["rewards"].data)
        data = self.memory.get_trajectory()

        transitions = {
            "o_0": np.array(data["states"]),
            "r_0": data["rewards"],
            "u_0": np.array(data["action"]),
            "log_prob": np.array(data["a_logit"]),
        }

        obs = self.tensor_to_cuda(torch.tensor(transitions["o_0"], dtype=torch.float))
        action = self.tensor_to_cuda(torch.tensor(transitions["u_0"], dtype=torch.long).view(-1, 1))
        reward = transitions["r_0"]
        old_action_log_prob = self.tensor_to_cuda(torch.tensor(
            transitions["log_prob"], dtype=torch.float
        ).view(-1, 1))

        # 计算reward-to-go
        R = 0
        Gt = []
        for r in reward[::-1]:  # 反过来
            R = r[0] + self.gamma * R
            Gt.insert(0, R)
        Gt = self.tensor_to_cuda(torch.tensor(Gt, dtype=torch.float))

        training_results = {"policy_loss": [], "value_loss": []}

        for i in range(self.update_freq):
            for index in BatchSampler(
                SubsetRandomSampler(range(data_length)), self.batch_size, False
            ):
                Gt_index = Gt[index].view(-1, 1)
                if self.use_cnn_encoder:
                    critic_encoded_obs = self.critic_cnn_encoder(obs[index])
                    actor_encoded_obs = self.actor_cnn_encoder(obs[index])
                else:
                    critic_encoded_obs = obs[index]
                    actor_encoded_obs = obs[index]

                V = self.critic(critic_encoded_obs)
                delta = Gt_index - V
                advantage = delta.detach()


                action_prob = self.actor(actor_encoded_obs).gather(1, action[index])

                ratio = action_prob / old_action_log_prob[index]
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * advantage
                )

                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                if self.use_cnn_encoder:
                    nn.utils.clip_grad_norm_(self.actor_cnn_encoder.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                if self.use_cnn_encoder:
                    nn.utils.clip_grad_norm_(self.critic_cnn_encoder.parameters(), self.max_grad_norm)

                training_results['policy_loss'].append(action_loss.item())
                training_results['value_loss'].append(value_loss.item())

                self.critic_net_optimizer.step()
                self.training_step += 1
        self.memory.item_buffer_clear()

        for tag, v in training_results.items():
            training_results[tag] = np.mean(v)

        return training_results

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)
        if self.use_cnn_encoder:
            encoder_path = os.path.join(base_path, f"encoder_{episode}.pth")
            encoder_state_dict = {}
            encoder_state_dict['actor_cnn_encoder'] = self.actor_cnn_encoder.state_dict()
            encoder_state_dict['critic_cnn_encoder'] = self.critic_cnn_encoder.state_dict()
            torch.save(encoder_state_dict, encoder_path)

    def load(self, actor_net, critic_net):
        self.actor.load_state_dict(torch.load(actor_net))
        self.critic.load_state_dict(torch.load(critic_net))
