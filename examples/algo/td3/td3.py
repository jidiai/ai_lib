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
    #
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
