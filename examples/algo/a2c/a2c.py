'''
in progress

'''
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from helper.multiprocessing_env import SubprocVecEnv
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from Network import Actor, Critic
import argparse


# 支持并行
def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

def t_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def plot(frame_idx, rewards):
    plt.plot(rewards, 'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)

num_envs = 8
env_name = "CartPole-v0"
env = gym.make(env_name)  # a single env

class A2C():
    def __init__(self, state_space, action_space, args):

        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.gamma = args.gamma
        self.hidden_size = args.hidden_size

        self.actor_net = Actor(state_space, action_space, self.hidden_size)
        self.critic_net = Critic(state_space, 1, self.hidden_size)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.a_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.c_lr)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            dist = self.actor_net(state)
            action = self.critic_net(state)
        return dist, action

    def save(self, save_path):
        torch.save(self.actor_net.state_dict(), str(save_path) + '/actor_net.pth')
        torch.save(self.critic_net.state_dict(), str(save_path) + '/critic_net.pth')

    def rollout(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.FloatTensor(reward).unsqueeze(1))
        self.masks.append(torch.FloatTensor(1 - done).unsqueeze(1))

    def update(self, next_state, log_probs, rewards, masks, values, entropy):
        next_state = torch.FloatTensor(next_state)
        next_value = self.critic_net(next_state)
        returns = self.compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns

def main(args):
    device = 'cpu'

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)  # 8 env

    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n
    num_steps = 5
    agent = A2C(num_inputs, num_outputs, args)
    state = envs.reset()

    max_frames = 1000
    frame_idx = 0
    test_rewards = []
    while frame_idx < max_frames:

        entropy = 0

        # rollout trajectory
        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = agent.choose_action(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            agent.rollout(log_prob, value, reward, done)

            state = next_state
            frame_idx += 1

            if frame_idx % 100 == 0:
                test_rewards.append(np.mean([t_env() for _ in range(10)]))
                plot(frame_idx, test_rewards)
        agent.update(next_state, log_probs, rewards, masks, values, entropy)
    envs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)
    parser.add_argument('--max_episodes', default=500, type=int)
    parser.add_argument('--algo', default="ppo", type=str, help="dqn/ppo/a2c")
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--hidden_size', default=256)
    args = parser.parse_args()
    main(args)
    print("end")