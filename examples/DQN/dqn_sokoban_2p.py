from IPython import display
import matplotlib.pyplot as plt

import gym
import math
import random

import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
# TODO 路径
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from env.chooseenv import make
from tensorboardX import SummaryWriter
import argparse


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class DQN(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # hyper paras #TODO
        self.hidden_dim = 32
        self.lr = 0.001
        self.capacity = 1280
        self.batch_size = 64
        self.gamma = 0.8

        self.critic = Network(self.state_dim, self.hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.buffer = []
        self.steps = 0
        self.learn_times = 0

        # TODO exploration
        # self.eps_low = 0.0
        # self.eps_high = 1.0
        # self.eps_decay = 0.99
        self.eps_fix = 0.1

        # TODO 保存
        self.game_name = game_name

        model_dir = Path('./models') / self.game_name
        if not model_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        para_dir = run_dir / 'params'
        os.makedirs(log_dir)
        os.makedirs(para_dir)
        self.writer = SummaryWriter(str(log_dir))
        self.para_dir = para_dir

        # TODO
        self.train = True

    def select_action(self, observation):
        if self.train:
            self.steps += 1
            # eps = self.eps_low + (self.eps_high - self.eps_low) * (math.exp(-1.0 * self.steps / self.eps_decay))
            eps = self.eps_fix
            if random.random() < eps:
                action = random.randrange(self.action_dim)
            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                action = torch.argmax(self.critic(observation)).item()
            return action
        else:
            observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.critic(observation)).item()
            return action

    def store_transition(self, obs, action, reward, obs_):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append([obs, action, reward, obs_])

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        self.learn_times += 1
        samples = random.sample(self.buffer, self.batch_size)
        obs, action, reward, obs_ = zip(*samples)
        obs = torch.tensor(obs, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1)
        obs_ = torch.tensor(obs_, dtype=torch.float)

        q_pred = reward + self.gamma * torch.max(self.critic(obs_).detach(), dim=1)[0].view(self.batch_size, -1)
        q_current = self.critic(obs).gather(1, action)

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_pred, q_current)
        self.writer.add_scalar('sokoban/loss', loss, self.learn_times)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.critic.state_dict(),  './' + str(self.para_dir) + '/critic_net.pth')
        # torch.save(self.optimizer.state_dict(), 'model_checkpoint.optimizer')

    def load(self, file):
        self.critic.load_state_dict(torch.load(file))
        # self.optimizer.load_state_dict(torch.load('model_checkpoint.optimizer'))

def RL_train(times, env):
    score = []
    for i in range(times):
        obs = env.reset()
        joint_action = []
        steps = 0
        reward_tot = 0
        for step in range(env.max_step):
            env._render()
            obs_ = state_wrapper(obs)
            for n in range(game.n_player):
                joint_action.append(agent.select_action(obs_[0:64] + obs_[64+n]))
            joint_action_ = action_wrapper(joint_action)
            #TODO reward
            obs_next, reward, done, info_before, info_after = env.step(joint_action_)
            obs_next_ = state_wrapper(obs_next)
            for n in range(env.n_player):
                agent.store_transition(obs_[0:64]+obs_[64+n],
                                       joint_action[n],
                                       reward[n],
                                       obs_next_[0:64]+obs_next_[64+n])
                agent.learn()
            obs = obs_next

            joint_action = []
            reward_tot += reward[0]
            steps += 1

            if env.is_terminal():
                break
        agent.writer.add_scalar('sokoban/return', reward_tot, i)
        agent.writer.add_scalar('sokoban/steps', steps, i)
        score.append(reward_tot)
        print('train time: ', i, 'reward_tot: ', reward_tot, 'steps: ', steps)
    agent.save()
    # plot(score)

def RL_evaluate(times, env):
    score = []
    for i in range(times):
        obs = env.reset()
        joint_action = []
        steps = 0
        reward_tot = 0
        for step in range(env.max_step):
            env._render()
            time.sleep(2)
            obs_ = state_wrapper(obs)
            for n in range(game.n_player):
                joint_action.append(agent.select_action(obs_[0:64] + obs_[64+n]))
            joint_action_ = action_wrapper(joint_action)
            #TODO reward
            obs_next, reward, done, info_before, info_after = env.step(joint_action_)
            obs = obs_next

            joint_action = []
            reward_tot += reward[0]
            steps += 1

            # if env.is_terminal():
            #     break
        score.append(reward_tot)
        print('train time: ', i, 'reward_tot: ', reward_tot, 'steps: ', steps)
    print('average score: ', np.mean(np.array(score)))

def state_wrapper(obs):
    '''
    :param state:
    :return: wrapped state
    '''
    obs_ = []
    for i in range(game.board_height):
        for j in range(game.board_width):
            obs_.append(obs[i][j][0])
    for n in range(game.n_player):
        obs_.append([n+4])
    return obs_

def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    for a in range(game.n_player):
        action_a = joint_action[a]
        each = [0] * game.action_dim
        each[action_a] = 1
        action_one_hot = [[each]]
        joint_action_.append([action_one_hot[0][0]])
    return joint_action_

def plot(score):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(20, 10))
    plt.clf() # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.show()

if __name__ == '__main__':

    game_name = "sokoban_2p"
    game = make(game_name)
    action_dim = game.action_dim
    state_dim = game.input_dimension
    # TODO
    state_dim_wrapped = state_dim + 1
    print('game.n_player', game.n_player)
    print('game.agent_nums', game.agent_nums)
    print('action_dim', action_dim, 'input_dim', state_dim, 'input_dim_wrapped', state_dim_wrapped)
    agent = DQN(state_dim_wrapped, action_dim)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--time", help="Name of environment", default="simple_push")
    # parser.add_argument("--model_name",
    #                     help="Name of directory to store " +
    #                          "model/training contents", default="./model")

    # TODO hyper: param
    times = 10
    agent.train = False

    if agent.train:
        RL_train(times, game)
    else:
        # TODO load
        agent.load('./models/sokoban_2p/run2/params/critic_net.pth')
        RL_evaluate(times, game)

    '''
    env = gym.make('CartPole-v0')
    params = {
        'gamma': 0.8,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    agent = DQN(params['state_space_dim'], params['action_space_dim'])

    score = []
    mean = []

    for episode in range(500):
        s0 = env.reset()
        total_reward = 1
        while True:
            # env.render()
            a0 = agent.select_action(s0)

            s1, r1, done, _ = env.step(a0)

            if done:
                r1 = -1

            agent.store_transition(s0, a0, r1, s1)

            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()
        print('total_reward', total_reward)
        score.append(total_reward)
        mean.append(sum(score[-100:]) / 100)

    # plot(score, mean)
    '''

