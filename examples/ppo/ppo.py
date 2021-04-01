import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

from Network import Actor, Critic

# Parameters
render = False
seed = 1
log_interval = 10

env = gym.make('CartPole-v0').unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print('num_state', state_dim, 'num_action', action_dim) # 4, 2
torch.manual_seed(seed)
env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class PPO():
    # clip_param = 0.2
    # max_grad_norm = 0.5
    # ppo_update_time = 10
    # buffer_capacity = 1000
    # batch_size = 32

    def __init__(self, state_dim, action_dim, args):

        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.ppo_update_time = args.ppo_update_time
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.gamma = args.gamma

        self.actor_net = Actor(state_dim, action_dim)
        self.critic_net = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.a_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.c_lr)

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        # self.writer = SummaryWriter('../exp')
        # if not os.path.exists('../param'):
        #     os.makedirs('../param/net_param')
        #     os.makedirs('../param/img')

    def choose_action(self, state):
        # print('1 state', state.shape)
        state = torch.from_numpy(state).float().unsqueeze(0) #  numpy -> tensor
        # print('!!!!! state shape', state.shape)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        # 如果tensor只有一个元素那么调用item方法的时候就是将tensor转换成python的scalars
        # print('action_prob[:, action.item()]',action_prob.size(), action_prob[:, action.item()].shape)
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save(self, save_path):
        torch.save(self.actor_net.state_dict(), str(save_path) + '/actor_net.pth')
        torch.save(self.critic_net.state_dict(), str(save_path) + '/critic_net.pth')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # print('----------', len(reward))
        # print('#', len(reward[::-1]))
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        # 计算reward-to-go
        R = 0
        Gt = []
        for r in reward[::-1]: # 反过来
            R = r + self.gamma * R
            Gt.insert(0, R)
            # print('Gt', Gt)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                # print('index', len(index)) # 为什么有的时候是7 有的时候是32
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                # print('self.actor_net(state[index])', self.actor_net(state[index]).shape) # 32, 2
                # print('action_prob', self.actor_net(state[index]).gather(1, action[index]).shape) # 32, 1
                # print('--self.actor_net(state[index])', self.actor_net(state[index]))  # 32, 2
                # print('--action_prob', self.actor_net(state[index]).gather(1, action[index])) # 32, 1
                # 根据维度dim开始查找 eg. a.gather(0,b) dim=0
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience

    def load(self, actor_net, critic_net):
        self.actor_net.load_state_dict(torch.load(actor_net))
        self.critic_net.load_state_dict(torch.load(critic_net))

def main(args):
    agent = PPO(state_dim, action_dim,args)
    for i_epoch in range(1000):
        state = env.reset()
        if render: env.render()

        for t in count():
            action, action_prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state

            if done :
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)
    parser.add_argument('--max_episodes', default=1000, type=int)
    parser.add_argument('--algo', default="ppo", type=str, help="dqn/ppo/a2c")

    parser.add_argument('--buffer_capacity', default=int(1e5), type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    args = parser.parse_args()

    main(args)
    print("end")