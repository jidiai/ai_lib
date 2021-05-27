import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import random
from env.chooseenv import make
from torch.distributions import Normal, Categorical
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 100)
        self.action_head = nn.Linear(100, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    # clip_param = 0.2
    # max_grad_norm = 0.5
    # ppo_update_time = 10
    # buffer_capacity = 1000
    # batch_size = 32

    def __init__(self, state_dim, action_dim):

        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_update_time = 10
        self.buffer_capacity = 1000
        self.batch_size = 32
        self.a_lr = 0.0001
        self.c_lr = 0.0001
        self.gamma = 0.99

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
        # print('self.buffer', self.buffer)
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
            R = r + gamma * R
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

action_dim = 2
state_dim = 4
env_type = "classic_CartPole-v0"
env = make(env_type, conf=None)
agent = PPO(state_dim, action_dim)
actor_net = os.path.dirname(os.path.abspath(__file__)) + '/actor_net.pth'
critic_net = os.path.dirname(os.path.abspath(__file__)) + '/critic_net.pth'
agent.load(actor_net, critic_net)

def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    for a in range(env.n_player):
        action_a = joint_action[a]
        each = [0] * env.action_dim
        each[action_a] = 1
        action_one_hot = [[each]]
        joint_action_.append([action_one_hot[0][0]])
    return joint_action_

def my_controller(obs_list, action_space_list, obs_space_list):
    try:
        action, _ = agent.choose_action(np.array(obs_list[0]))
    except:
        action, _ = agent.choose_action(np.array(obs_list[0][0]))
    action_ = action_wrapper([action])
    return action_