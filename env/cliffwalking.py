# -*- coding:utf-8  -*-
# Time  : 2021/8/13 下午3:01
# Author: Yahui Cui

from env.simulators.game import Game
from utils.discrete import Discrete

import json
import numpy as np


class CliffWalking(Game):
    def __init__(self, conf):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.env_core = CliffWalkingEnv(12, 4)

        self.load_action_space(conf)

        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])
        self.won = {}
        self.joint_action_space = self.set_action_space()
        self.n_return = [0] * self.n_player

        self.action_dim = self.get_action_dim()
        self.input_dimension = self.env_core.nrow * self.env_core.ncol
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
        obs, reward, self.done = self.env_core.step(action)
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

        return self.done

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


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0 # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1 # 记录当前智能体位置的纵坐标

    def step(self, action): # 外部调用这个函数来让当前位置改变
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 4 种动作, 0:上, 1:下, 2:左, 3:右。原点(0,0)定义在左上角
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0: # 下一个位置在悬崖或者终点
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self): # 回归初始状态，坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
