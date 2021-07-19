import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np

from algo.duelingq.Network import Critic

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action"]


# 定义Dueling DQN类（定义两个网络）
class DUELINGQ(object):
    # 定义Dueling DQN的一系列属性
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.hidden_size = args.hidden_size
        self.lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.critic_eval = Critic(self.state_dim,  self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optimizer.Adam(self.critic_eval.parameters(), lr=self.lr)

        # exploration
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episode * 100)

        # 更新target网
        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

    # 定义动作选择函数（x为状态）
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

    # 定义学习函数（记忆库已满后便开始学习）
    def learn(self):

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
        reward = torch.tensor(transitions["r_0"], dtype=torch.float).squeeze()
        done = torch.tensor(transitions["d_0"], dtype=torch.float).squeeze()

        q_eval = self.critic_eval(obs).gather(1, action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0] * (1 - done)).view(self.batch_size, 1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        return loss

    def save(self):
        torch.save(self.critic_eval.state_dict(), 'critic_net.pth')

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))