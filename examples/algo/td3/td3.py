# import argparse
# from collections import namedtuple
# from itertools import count
#
# import os, sys, random
# import numpy as np
#
# import gym
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Normal
# from tensorboardX import SummaryWriter
#
# import mlagents
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# parser.add_argument("--env_name", default="classic_Pendulum-v0")  # OpenAI gym environment name， BipedalWalker-v2
# parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
# parser.add_argument('--target_update_interval', default=1, type=int)
# parser.add_argument('--iteration', default=10, type=int)
#
# parser.add_argument('--learning_rate', default=3e-4, type=float)
# parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
# parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
# parser.add_argument('--num_iteration', default=20000, type=int) #  num of  games
# parser.add_argument('--batch_size', default=100, type=int) # mini batch size
# parser.add_argument('--seed', default=1, type=int)
#
# # optional parameters
# parser.add_argument('--num_hidden_layers', default=2, type=int)
# parser.add_argument('--sample_frequency', default=256, type=int)
# parser.add_argument('--activation', default='Relu', type=str)
# parser.add_argument('--render', default=False, type=bool) # show UI or not
# parser.add_argument('--log_interval', default=50, type=int) #
# parser.add_argument('--load', default=False, type=bool) # load model
# parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
# parser.add_argument('--policy_noise', default=0.2, type=float)
# parser.add_argument('--noise_clip', default=0.5, type=float)
# parser.add_argument('--policy_delay', default=2, type=int)
# parser.add_argument('--exploration_noise', default=0.1, type=float)
# parser.add_argument('--max_episode', default=2000, type=int)
# parser.add_argument('--print_log', default=5, type=int)
# args = parser.parse_args()
#
# from pathlib import Path
# import sys
# base_dir = Path(__file__).resolve().parent.parent.parent.parent
# sys.path.append(str(base_dir))
# from env.chooseenv import make

# # Set seeds
# # env.seed(args.seed)
# # torch.manual_seed(args.seed)
# # np.random.seed(args.seed)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# script_name = os.path.basename(__file__)
# env = make(args.env_name)
#
# # state_dim = env.observation_space.shape[0]
# # action_dim = env.action_space.shape[0]
# state_dim = env.input_dimension.shape[0]
# action_dim = env.action_dim.shape[0]
# max_action = float(env.action_dim.high[0])
# print(state_dim, action_dim, max_action)
# min_Val = torch.tensor(1e-7).float().to(device) # min value
#
# directory = './exp' + script_name + args.env_name +'./'
# '''
# Implementation of td3 with pytorch
# Original paper: https://arxiv.org/abs/1802.09477
# Not the author's implementation !
# '''


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# from algo.dqn.Network import Critic

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer


def get_trajectory_property():
    return ["action"]

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        # todo: self.max_action

        self.hidden_size = args.hidden_size
        # self.c_lr = args.c_lr
        # self.a_lr = args.a_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau


        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action)
        self.critic_1 = Critic(self.state_dim, self.action_dim)
        self.critic_1_target = Critic(self.state_dim, self.action_dim)
        self.critic_2 = Critic(self.state_dim, self.action_dim)
        self.critic_2_target = Critic(self.state_dim, self.action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters()) # ？？？？
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # self.memory = Replay_buffer(args.capacity)
        # self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float()
        return self.actor(state).cpu().data.numpy().flatten()
    '''
    def update(self, num_iteration):

        if self.num_training % 500 == 0:

        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(d)
            reward = torch.FloatTensor(r)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1
    '''
    # def save(self):
    #     torch.save(self.actor.state_dict(), directory+'actor.pth')
    #     torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
    #     torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
    #     torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
    #     torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
    #     torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
    #     print("====================================")
    #     print("Model has been saved...")
    #     print("====================================")

    # def load(self):
    #     self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
    #     self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
    #     self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
    #     self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
    #     self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
    #     self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
    #     print("====================================")
    #     print("model has been loaded...")
    #     print("====================================")

'''
def main():
    agent = TD3(state_dim, action_dim, max_action)
    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            state = env.reset()
            ep_r = 0
            for t in count():
                state = np.array(state)
                action = agent.select_action(state)
                next_state, reward, done, info, _ = env.step(np.float32([action]))
                ep_r += reward[0]
                # env.render()
                if done or t ==2000 :
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        ep_r = 0
        for i in range(args.num_iteration):
            state = env.reset()
            for t in range(2000):
                state = np.array(state)
                action = agent.select_action(state)
                action = action + np.random.normal(0, args.exploration_noise, size=env.action_dim.shape[0])
                action = action.clip(env.action_dim.low, env.action_dim.high)
                next_state, reward, done, info, _ = env.step([action])
                ep_r += reward[0]
                if args.render and i >= args.render_interval : env.render()
                agent.memory.push((state, next_state, action, reward, np.float(done)))
                if i+1 % 10 == 0:
                    print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                if len(agent.memory.storage) >= args.capacity-1:
                    agent.update(10)

                state = next_state
                if done or t == args.max_episode -1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
'''