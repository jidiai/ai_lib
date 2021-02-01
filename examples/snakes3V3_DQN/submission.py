import os
from env.chooseenv import make
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


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
        self.hidden_dim = 64
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

        # TODO
        self.train = False

    def choose_action(self, observation):
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

    def load(self, file):
        self.critic.load_state_dict(torch.load(file))


def state_wrapper(obs):
    '''
    :param state:
    :return: wrapped state
    '''
    obs_ = []
    for i in range(game.board_height):
        for j in range(game.board_width):
            obs_.append(obs[i][j][0])
    # TODO hard code
    for n in game.snakes_position:
        obs_.append(game.snakes_position[n][0][0])
        obs_.append(game.snakes_position[n][0][1])
    return obs_


def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    for a in range(3):
        action_a = joint_action[a]
        each = [0] * game.action_dim
        each[action_a] = 1
        action_one_hot = [[each]]
        joint_action_.append([action_one_hot[0][0]])
    return joint_action_


game_name = "snakes_3v3"
game = make(game_name)
action_dim = game.action_dim
state_dim = game.input_dimension
state_dim_wrapped = state_dim + 2
agent = DQN(state_dim_wrapped, action_dim)
model_path = os.path.dirname(os.path.abspath(__file__)) + '/critic_net.pth'
agent.load(model_path)


def my_controller(observation_list, action_space_list, obs_space_list):
    joint_action = []
    obs_ = state_wrapper(observation_list[0])
    for n in range(3):
        joint_action.append(agent.choose_action(obs_[0:200] + obs_[(200+n):(200+n+2)]))
    joint_action_ = action_wrapper(joint_action)
    return joint_action_