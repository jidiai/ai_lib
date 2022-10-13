# -*- coding:utf-8  -*-
# Time  : 2021/8/9 下午3:45
# Author: Yahui Cui


from env.simulators.game import Game
from utils.discrete import Discrete

import json
import numpy as np


W = -100  # wall
G = 100  # goal
GRID_LAYOUT = np.array([
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, 0, W, W, W, W, W, W, 0, W, W],
    [W, 0, 0, 0, 0, 0, 0, 0, 0, G, 0, W],
    [W, 0, 0, 0, W, W, W, W, 0, 0, 0, W],
    [W, 0, 0, 0, W, W, W, W, 0, 0, 0, W],
    [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W],
    [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W],
    [W, W, 0, 0, 0, 0, 0, 0, 0, 0, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W]
])


class GridWorld(Game):
    def __init__(self, conf):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.env_core = Grid()

        self.load_action_space(conf)

        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])
        self.won = {}
        self.joint_action_space = self.set_action_space()
        self.n_return = [0] * self.n_player

        self.action_dim = self.get_action_dim()
        self.input_dimension = self.env_core.number_of_states
        self.init_info = None
        self.reset()

    def load_action_space(self, conf):
        if "act_box" in conf:
            input_action = json.loads(conf["act_box"]) if isinstance(conf["act_box"], str) else conf["act_box"]
            # print(input_action)
            if "discrete_n" not in input_action:
                raise Exception("act_box in discrete case must have field discrete_n")
            discrete_n = int(input_action["discrete_n"])
            self.env_core.action_space = Discrete(discrete_n)

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        action = self.decode(joint_action)
        info_before = self.step_before_info()
        info_after = {}
        # print("action in step ", action)
        reward, discount, obs = self.env_core.step(action)
        info_after['discount'] = discount
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()[0]
        reward = self.get_reward(reward)
        self.current_state = [obs] * self.n_player
        self.all_observes = self.get_all_observes()
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def decode(self, joint_action):
        return joint_action[0][0].index(1)

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                raise Exception("The input action dimension for player {} should be {}, not {}".format(
                    i, self.joint_action_space[i][0].n, len(joint_action[i][0])))

    def get_reward(self, reward):
        r = [0] * self.n_player
        # print("reward is ", reward)
        for i in range(self.n_player):
            r[i] = reward
            self.n_return[i] += r[i]
        return r

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done or self.env_core._done

    def set_action_space(self):
        action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
        return action_space

    def check_win(self):
        return '0'

    def reset(self):
        observation = self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        self.current_state = [observation] * self.n_player
        self.all_observes = self.get_all_observes()
        return self.all_observes

    def get_action_dim(self):
        action_dim = 1

        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_all_observes(self):
        all_observes = []
        for i in range(len(self.current_state)):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)
        return all_observes


class Grid(object):

    def __init__(self, noisy=False):
        # -1: wall
        # 0: empty, episode continues
        # other: number indicates reward, episode will terminate
        self._layout = GRID_LAYOUT
        self._start_state = (2, 2)
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._noisy = noisy
        self._done = False

    @property
    def number_of_states(self):
        return self._number_of_states

    def reset(self):
        self._layout = GRID_LAYOUT
        self._start_state = (2, 2)
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._done = False
        return self.get_obs()

    def get_obs(self):
        y, x = self._state
        return y * self._layout.shape[1] + x

    def obs_to_state(self, obs):
        x = obs % self._layout.shape[1]
        y = obs // self._layout.shape[1]
        s = np.copy(self._layout)
        s[y, x] = 4
        return s

    def step(self, action):
        y, x = self._state

        if action == 0:  # up
            new_state = (y - 1, x)
        elif action == 1:  # right
            new_state = (y, x + 1)
        elif action == 2:  # down
            new_state = (y + 1, x)
        elif action == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

        new_y, new_x = new_state
        reward = self._layout[new_y, new_x]
        reward = float(reward)
        if self._layout[new_y, new_x] == W:  # wall
            discount = 0.9
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = -1.
            discount = 0.9
        else:  # a goal
            self._done = True
            discount = 0.
            new_state = self._start_state

        if self._noisy:
            width = self._layout.shape[1]
            reward += 10 * np.random.normal(0, width - new_x + new_y)

        self._state = new_state
        return reward, discount, self.get_obs()
