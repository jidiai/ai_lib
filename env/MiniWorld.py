# -*- coding:utf-8  -*-
# Time  : 2021/02/28 16:33
# Author: Xue Yan

from env.simulators.game import Game
from env.obs_interfaces.observation import *
import numpy as np
import json
from utils.discrete import Discrete
from utils.box import Box
import gym
import gym_miniworld


class MiniWorld(Game, VectorObservation):
    def __init__(self, conf):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])
        self.env_core = gym.make(self.game_name)
        self.load_action_space(conf)

        observation = self.env_core.reset()
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        obs_list = observation.reshape(-1).tolist()

        self.won = {}
        self.current_state = [obs_list] * self.n_player
        self.all_observes = self.get_all_observes()
        self.n_return = [0] * self.n_player
        self.joint_action_space = self.set_action_space()

        self.action_dim = self.get_action_dim()
        self.input_dimension = self.env_core.observation_space

        self.ob_space = [self.env_core.observation_space for _ in range(self.n_player)]#60* 80 *3
        self.ob_vector_shape = [self.env_core.observation_space.shape] * self.n_player
        self.ob_vector_range = [self.env_core.observation_space.low,
                                self.env_core.observation_space.high] * self.n_player#???
        self.init_info = None

    def load_action_space(self, conf):
        if "act_box" in conf:
            input_action = json.loads(conf["act_box"]) if isinstance(conf["act_box"], str) else conf["act_box"]
            # print(input_action)
            if self.is_act_continuous:
                if ("high" not in input_action) or ("low" not in input_action) or ("shape" not in input_action):
                    raise Exception("act_box in continuous case must have fields low, high, shape")
                shape = tuple(input_action["shape"])
                self.env_core.action_space = Box(input_action["low"], input_action["high"], shape, np.float32)
            else:
                if "discrete_n" not in input_action:
                    raise Exception("act_box in discrete case must have field discrete_n")
                discrete_n = int(input_action["discrete_n"])
                self.env_core.action_space = Discrete(discrete_n)

    def get_next_state(self, action):#action=0/1/2
        observation, reward, done, info = self.env_core.step(action)

        return observation, reward, done, info

    def set_action_space(self):
        if self.is_act_continuous:
            action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
        else:
            action_space = [[self.env_core.action_space] for _ in range(self.n_player)]#discrete(3)
        return action_space

    def step(self, joint_action):
        action = self.decode(joint_action)
        info_before = self.step_before_info()
        # print("action in step ", action)
        next_state, reward, self.done, info_after = self.get_next_state(action)
        # self.current_state = next_state
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()[0]
        reward = self.get_reward(reward)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        next_state = next_state.reshape(-1).tolist()
        self.current_state = [next_state] * self.n_player
        self.all_observes = self.get_all_observes()
        done = self.is_terminal()
        info_after = self.parse_info(info_after)
        self.step_cnt += 1
        return self.all_observes, reward, done, info_before, info_after

    def get_reward(self, reward):
        r = [0] * self.n_player
        # print("reward is ", reward)
        for i in range(self.n_player):
            r[i] = reward
            self.n_return[i] += r[i]
        return r

    def decode(self, joint_action):

        if not self.is_act_continuous:
            return joint_action[0][0].index(1)#？？
        else:
            return joint_action[0]

    def step_before_info(self, info=''):
        return info

    def parse_info(self, info):
        new_info = {}
        for key, val in info.items():
            if isinstance(val, np.ndarray):
                new_info[key] = val.tolist()
            else:
                new_info[key] = val
        return new_info

    def is_terminal(self):
        if self.step_cnt > self.max_step:
            self.done = True

        return self.done

    def check_win(self):
        if self.env_core.near(self.env_core.box):
            return 1
        else:
            return -1

    def reset(self):
        observation = self.env_core.reset()
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        obs_list = observation.reshape(-1).tolist()
        self.step_cnt = 0
        self.done = False
        self.current_state = [obs_list] * self.n_player
        self.all_observes = self.get_all_observes()
        return self.all_observes

    def get_action_dim(self):
        action_dim = 1
        print("joint action space is ", self.joint_action_space[0][0])
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_vector_obs_config(self, player_id):
        return self.ob_vector_shape[player_id], self.ob_vector_range[player_id]

    def get_vector_many_obs_space(self, player_id_list):
        all_obs_space = {}
        for i in player_id_list:
            m = self.ob_vector_shape[i]
            all_obs_space[i] = m
        return all_obs_space

    def get_vector_observation(self, current_state, player_id, info_before):
        return self.current_state[player_id]

    def get_render_data(self, current_state):
        return []

    def set_seed(self, seed=None):
        self.env_core.seed(seed)

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)
        return all_observes
