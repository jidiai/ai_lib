# -*- coding:utf-8  -*-
# Time  : 2021/03/03 16:52
# Author: Yutong Wu

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from Network import Actor, Critic

import argparse
from itertools import count
from collections import namedtuple

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make

class DDPG(object):
    def __init__(self, state_space, action_space, args):
        self.action_space = action_space
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device
        self.update_freq = args.update_freq
        self.epsilon = args.epsilon

        self.actor = Actor(state_space, action_space).to(self.device)
        self.actor_target = Actor(state_space, action_space).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_space, action_space).to(self.device)
        self.critic_target = Critic(state_space, action_space).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = []

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float).to(self.device)
        a = self.actor(s).cpu().squeeze(0).detach().numpy()
        return a

    def store_transition(self, transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):

        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)

        for _ in range(self.update_freq):
            s0, a0, r1, s1, done = zip(*samples)

            s0 = torch.tensor(s0, dtype=torch.float).squeeze(1).to(self.device)
            a0 = torch.tensor(a0, dtype=torch.float).to(self.device)
            r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(self.device)
            s1 = torch.tensor(s1, dtype=torch.float).squeeze(1).to(self.device)
            done = torch.tensor(done, dtype=torch.float).view(self.batch_size, -1).to(self.device)

            with torch.no_grad():
                a1 = self.actor_target(s1)
                y_true = r1 + self.gamma * (1 - done) * self.critic_target(s1, a1)
            print(a0)
            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

            self.soft_update(self.critic_target, self.critic, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)

    def soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, save_path, i):
        torch.save(self.actor.state_dict(), str(save_path) + '/actor_net_{}.pth'.format(i))

    def load(self, load_path, i):
        self.actor.load_state_dict(torch.load(str(load_path) + '/actor_net_{}.pth'.format(i)))

    def scale_noise(self, decay):
        self.actor.noise.scale = self.actor.noise.scale * decay

def logits2action(logits):
    print(logits)
    # action = m.sample().item()
    action = np.argmax(logits)
    return action

def main(args):
    global env
    env = make(args.scenario)
    action_space = env.action_dim
    observation_space = env.input_dimension.shape[0]
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    agent = DDPG(observation_space, action_space, args)

    for i_epoch in range(10000):
        state = env.reset()
        Gt = 0
        for t in count():
            state = np.array(state)
            logits = agent.choose_action(state)
            action = logits2action(logits)
            action_ = action_wrapper([action])
            next_state, reward, done, _, _ = env.step(action_)
            reward = np.array(reward)

            trans = Transition(state, probs.numpy(), reward, np.array(next_state), done)

            if args.render:
                env.render()

            agent.store_transition(trans)
            state = next_state

            if done:
                reward = -1

            Gt += reward
            if done:
                print('i_epoch: ', i_epoch, 'Gt: ', Gt)
                if len(agent.buffer) >= agent.batch_size:
                    agent.learn()
                break

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)
    parser.add_argument('--max_episodes', default=500, type=int)
    parser.add_argument('--algo', default="ppo", type=str, help="dqn/ppo/a2c")

    parser.add_argument('--buffer_capacity', default=int(1024), type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.005, type=float)
    parser.add_argument('--c_lr', default=0.005, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--hidden_size', default=100)
    parser.add_argument('--epsilon', default=0.1)
    parser.add_argument('--max_episode', default=1000, type=int)
    parser.add_argument('--target_replace', default=100)
    parser.add_argument('--tau', default=0.02)

    parser.add_argument('--render', default=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--update_freq', default=5)

    args = parser.parse_args()

    main(args)
    print("end")
'''
# cartpole
if __name__ == '__main__':
    game_name = 'Pendulum-v0'
    env = gym.make(game_name)
    env.reset()
    # env.render()

    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
    }

    agent = Agent(**params)
    run_dir, log_dir = make_logpath(game_name)
    writer = SummaryWriter(str(log_dir))

    for episode in range(2000):
        s0 = env.reset()
        episode_reward = 0

        for step in range(200):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0, a0, r1, s1)

            episode_reward += r1
            s0 = s1

            agent.learn()

        print(episode, ': ', episode_reward)
'''