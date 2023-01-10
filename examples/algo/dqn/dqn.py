import random
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np

from examples.networks.critic import Critic
from examples.networks.encoder import CNNEncoder


import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from examples.common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action"]


class DQN(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.hidden_size = args.hidden_size
        self.lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        if 'max_grad_norm' not in vars(args):
            self.max_grad_norm = 0.1
        else:
            self.max_grad_norm = args.max_grad_norm

        self.use_cuda = args.use_cuda
        if 'cnn' in vars(args):
            self.use_cnn_encoder = True
            cnn_cfg = args.cnn
            self.cnn_encoder = CNNEncoder(input_chanel=cnn_cfg['input_chanel'],
                                          hidden_size=128,
                                          output_size=self.hidden_size,
                                          channel_list=cnn_cfg['channel_list'],
                                          kernel_list=cnn_cfg['kernel_list'],
                                          stride_list=cnn_cfg['stride_list'])
            self.critic_eval = Critic(
                input_size=self.hidden_size,
                output_size=self.action_dim,
                hidden_size=self.hidden_size,
                num_hidden_layer=args.num_hidden_layer
            )
            self.critic_target = Critic(
                input_size=self.hidden_size,
                output_size=self.action_dim,
                hidden_size=self.hidden_size,
                num_hidden_layer=args.num_hidden_layer
            )
            self.to_cuda()
            self.optimizer=optimizer.Adam(list(self.critic_eval.parameters())+list(self.cnn_encoder.parameters()), lr=self.lr)

        else:
            self.use_cnn_encoder = False
            self.critic_eval = Critic(
                self.state_dim,
                self.action_dim,
                self.hidden_size,
                num_hidden_layer=args.num_hidden_layer,
            )
            self.critic_target = Critic(
                self.state_dim,
                self.action_dim,
                self.hidden_size,
                num_hidden_layer=args.num_hidden_layer,
            )
            self.to_cuda()

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

    def to_cuda(self):
        if self.use_cuda:
            self.critic_eval.to('cuda')
            self.critic_target.to('cuda')
            if self.use_cnn_encoder:
                self.cnn_encoder.to('cuda')

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
        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                action = random.randrange(self.action_dim)

            else:

                observation = self.tensor_to_cuda(torch.tensor(observation, dtype=torch.float))
                if self.use_cnn_encoder:
                    observation = self.cnn_encoder(observation.unsqueeze(0))
                else:
                    observation = observation.view(1, -1)

                action = torch.argmax(self.critic_eval(observation)).item()
        else:
            observation = self.tensor_to_cuda(torch.tensor(observation, dtype=torch.float))
            if self.use_cnn_encoder:
                observation = self.cnn_encoder(observation.unsqueeze(0))
            else:
                observation = observation.view(1, -1)

            action = torch.argmax(self.critic_eval(observation)).item()

        return {"action": action}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self, **kwargs):
        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return {}

        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data["states"]),
            "o_next_0": np.array(data["states_next"]),
            "r_0": np.array(data["rewards"]).reshape(-1, 1),
            "u_0": np.array(data["action"]),
            "d_0": np.array(data["dones"]).reshape(-1, 1),
        }

        obs = self.tensor_to_cuda(torch.tensor(transitions["o_0"], dtype=torch.float))
        obs_ = self.tensor_to_cuda(torch.tensor(transitions["o_next_0"], dtype=torch.float))
        action = self.tensor_to_cuda(torch.tensor(transitions["u_0"], dtype=torch.long).view(
            self.batch_size, -1
        ))
        reward = self.tensor_to_cuda(torch.tensor(transitions["r_0"], dtype=torch.float).squeeze())
        done = self.tensor_to_cuda(torch.tensor(transitions["d_0"], dtype=torch.float).squeeze())

        if self.use_cnn_encoder:
            obs = self.cnn_encoder(obs)
            obs_ = self.cnn_encoder(obs_)

        q_eval = self.critic_eval(obs).gather(1, action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0] * (1 - done)).view(
            self.batch_size, 1
        )
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic_eval.parameters(), self.max_grad_norm)

        grad_dict = {}
        for name, param in self.critic_eval.named_parameters():
            grad_dict[f"Critic_eval/{name} gradient"] = param.grad.mean().item()
        if self.use_cnn_encoder:
            for name, param in self.cnn_encoder.named_parameters():
                grad_dict[f"CNN_encoder/{name} gradient"] = param.grad.mean().item()

        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        training_results = {"loss": loss.detach().cpu().numpy(),
                            "eps": self.eps}
        training_results.update(grad_dict)
        return training_results

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_eval.state_dict(), model_critic_path)

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))
