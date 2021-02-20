import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
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
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_dim = args.hidden_dim
        self.lr = args.lr
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.critic_eval = Network(self.state_dim, self.hidden_dim, self.action_dim)
        self.critic_target = Network(self.state_dim, self.hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.buffer = []

        self.game_name = args.game_name
        self.mode = args.mode

        self.eps_start = args.epsilon
        self.eps_end = 0.05
        self.eps_delay = 0.8 / args.max_episode
        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

    def select_action(self, observation, train=True):
        if train:
            eps = max(self.eps_end, self.eps_start - self.eps_delay)
            if random.random() < eps:
                action = random.randrange(self.action_dim)
            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                # print('observation', observation.size()) # 1, 213
                # print('self.critic(observation)', self.critic(observation).size()) # 1, 4
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
        self.critic.load_state_dict(torch.load(file))


def RL_evaluate(args):
    game_name = "snakes_3v3"
    global env
    env = make(game_name)
    action_dim = env.action_dim
    state_dim = env.input_dimension
    state_dim_wrapped = state_dim + 13
    players_id_list = range(0,3)

    agent = DQN(state_dim_wrapped, action_dim, args)

    obs = env.reset()
    obs_list = env.get_dict_many_observation(obs, players_id_list)
    for i in players_id_list:
        obs.append(get_observations(obs_list, i))
    joint_action = []
    steps = 0
    reward_tot = 0
    obs = []
    for step in range(env.max_step):
        # player 1
        for n in range(env.agent_nums[0]):
            joint_action.append(agent.select_action(obs[n], train=False))
        # player 2
        for n in range(env.agent_nums[1]):
            joint_action.append(np.random.randint(action_dim))

        joint_action_ = action_wrapper(joint_action)

        obs_next, reward, done, info_before, info_after = env.step(joint_action_)

        obs_next_list = env.get_dict_many_observation(obs_next, players_id_list)
        for i in players_id_list:
            obs_next.append(get_observations(obs_next_list, i))

        for n in range(env.agent_nums[0]):
            agent.store_transition(obs[n],
                                   joint_action[n],
                                   reward[n],
                                   obs_next[n])
        obs = obs_next

        joint_action = []
        reward_tot += np.sum(reward[0:3])
        steps += 1
    # writer.add_scalar(game_name + 'train/return', reward_tot, epi)

def get_random_1_person(action_space):
    joint_action = []
    for i in range(len(action_space)):
        player = []
        for j in range(len(action_space[i])):
            each = [0] * action_space[i][j]
            idx = np.random.randint(action_space[i][j])
            each[idx] = 1
            player.append(each)
        joint_action.append(player)
    return joint_action


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


def main(args):
    game_name = "snakes_3v3"
    global env
    env = make(game_name)
    action_dim = env.action_dim
    state_dim = env.input_dimension
    state_dim_wrapped = state_dim + 13
    players_id_list = range(0,3)

    agent = DQN(state_dim_wrapped, action_dim, args)

    score = []

    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / Path('./models') / game_name

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
    os.makedirs(log_dir)
    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(str(log_dir))

    for epi in range(args.max_episode):
        env.reset()
        obs_list = env.get_dict_many_observation(env.current_state, players_id_list)
        obs = []
        obs_next_ = []
        for i in players_id_list:
            obs.append(get_observations(obs_list, i))
        joint_action = []
        steps = 0
        reward_tot = 0
        for step in range(env.max_step):
            # player 1
            for n in range(env.agent_nums[0]):
                joint_action.append(agent.select_action(obs[n]))
            # player 2
            for n in range(env.agent_nums[1]):
                joint_action.append(np.random.randint(action_dim))

            joint_action_ = action_wrapper(joint_action)

            obs_next, reward, done, info_before, info_after = env.step(joint_action_)

            obs_next_list = env.get_dict_many_observation(obs_next, players_id_list)
            for i in players_id_list:
                obs_next_.append(get_observations(obs_next_list, i))

            for n in range(env.agent_nums[0]):
                agent.store_transition(obs[n], joint_action[n], reward[n], obs_next_[n])
            agent.learn()
            obs = obs_next_

            joint_action = []
            reward_tot += np.sum(reward[0:3])
            steps += 1

        score.append(reward_tot)
        print('train time: ', epi,
              'reward_tot: ', reward_tot,
              'average score %.2f' % np.mean(score[-100:]))

        writer.add_scalar(game_name + 'train/return', reward_tot, epi)

    agent.save(run_dir / 'model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set agent hyper parameter
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--buffer_size", default=1280, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--target_replace", default=100, type=int)

    parser.add_argument("--epsilon", default=1, type=float)

    parser.add_argument("--game_name", default= "snakes_3v3")
    parser.add_argument("--evaluate_rate", default=100, type=int)
    parser.add_argument("--max_episode", default=100, type=int)
    parser.add_argument("--tensorboard", default=True, type=bool)
    parser.add_argument("--mode", default="train", type=str, help="train/eval")
    parser.add_argument("--algo", default="DQN")
    parser.add_argument("--log_dir", default= "DQN")

    args = parser.parse_args()
    main(args)




