import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from env.chooseenv import make

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

        self.hidden_dim = 64
        self.lr = 0.001
        self.buffer_size = 1280
        self.batch_size = 64
        self.gamma = 0.8

        self.critic_eval = Network(self.state_dim, self.hidden_dim, self.action_dim)
        self.critic_target = Network(self.state_dim, self.hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.buffer = []

        self.eps_start = 1
        self.eps_end = 0.05
        self.eps_delay = 0.8 / 100
        self.learn_step_counter = 0
        self.target_replace_iter = 100

    def select_action(self, observation, train=True):
        if train:
            eps = max(self.eps_end, self.eps_start - self.eps_delay)
            if random.random() < eps:
                action = random.randrange(self.action_dim)
            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                action = torch.argmax(self.critic_eval(observation)).item()
        else:
            observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def store_transition(self, obs, action, reward, obs_):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([obs, action, reward, obs_])

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        obs, action, reward, obs_ = zip(*samples)
        obs = torch.tensor(obs, dtype=torch.float).squeeze()
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1).squeeze()
        obs_ = torch.tensor(obs_, dtype=torch.float).squeeze()

        q_eval = self.critic_eval(obs).gather(1,action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0]).view(self.batch_size,1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        return loss

    def save(self, save_path):
        torch.save(self.critic_eval.state_dict(),  save_path)

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))


def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    for a in range(3):
        action_a = joint_action[a]
        each = [0] * 4
        each[action_a] = 1
        action_one_hot = [[each]]
        joint_action_.append([action_one_hot[0][0]])
    return joint_action_

def get_observations(key_info, index):
    '''
    observation space: env.input_dimension + 6 * 2 (snake head) + 1 (index) = 213
    '''
    grid = [[[0] * env.cell_dim for _ in range(env.board_width)] for _ in range(env.board_height)]
    for key in key_info[0]:
        for pos in key_info[0][key]:
            grid[pos[0]][pos[1]] = [key]
    obs_ = []
    for i in range(env.board_height):
        for j in range(env.board_width):
            obs_.append(grid[i][j])
    for key in key_info[0]:
        if key > 1:
            obs_.append([key_info[0][key][0][0]])
            obs_.append([key_info[0][key][0][1]])
    obs_.append([index])
    return obs_

action_dim = 4
state_dim = 213
env_type = "snakes_3v3"
env = make(env_type, conf=None)
agent = DQN(state_dim, action_dim)
model_path = os.path.dirname(os.path.abspath(__file__)) + '/model.pth'
agent.load(model_path)
players_id_list = range(0,3)

def my_controller(obs_list, action_space_list, obs_space_list):
    obs = []
    for i in players_id_list:
        obs.append(get_observations(obs_list, i))

    joint_action = []
    for n in range(3):
        joint_action.append(agent.select_action(obs[n], train=False))
    joint_action_ = action_wrapper(joint_action)
    return joint_action_