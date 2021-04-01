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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from helper.multiprocessing_env import SubprocVecEnv


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


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


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def plot(frame_idx, rewards):
    plt.plot(rewards, 'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)


num_envs = 8
env_name = "CartPole-v0"
env = gym.make(env_name)  # a single env

def main():
    plt.ion()
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)  # 8 env

    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n

    # Hyper params:
    hidden_size = 256
    lr = 1e-3
    num_steps = 200
    global model
    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters())

    state = envs.reset()

    max_frames = 20000
    frame_idx = 0
    test_rewards = []
    while frame_idx < max_frames:

        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        # rollout trajectory
        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            frame_idx += 1

            if frame_idx % 100 == 0:
                test_rewards.append(np.mean([t_env() for _ in range(10)]))
                plot(frame_idx, test_rewards)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    envs.close()

if __name__ == '__main__':
    main()
    print("end")
