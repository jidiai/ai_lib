# -*- coding:utf-8  -*-
# Time  : 2020/12/28 16:33
# Author: Yahui Cui
from env.simulators.game import Game
from env.obs_interfaces.observation import *
import numpy as np
import json
from utils.discrete import Discrete
from utils.box import Box
import gym


class CCGame(Game, VectorObservation):
    def __init__(self, conf):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.env_core = gym.make(self.game_name)

        self.load_action_space(conf)
        observation = self.env_core.reset()
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        obs_list = observation.reshape(-1).tolist()

        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])
        self.won = {}
        # self.env_core.action_space = gym.spaces.Box(-4.0, 4.0, (1,), np.float32)
        self.joint_action_space = self.set_action_space()
        self.current_state = [obs_list] * self.n_player
        self.all_observes = self.get_all_observes()
        self.n_return = [0] * self.n_player

        self.action_dim = self.get_action_dim()
        self.input_dimension = self.env_core.observation_space
        self.ob_space = [self.env_core.observation_space for _ in range(self.n_player)]
        self.ob_vector_shape = [self.env_core.observation_space.shape] * self.n_player
        self.ob_vector_range = [self.env_core.observation_space.low, self.env_core.observation_space.high] * self.n_player
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

    def step(self, joint_action):
        self.is_valid_action(joint_action)
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
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def decode(self, joint_action):

        if not self.is_act_continuous:
            return joint_action[0][0].index(1)
        else:
            return joint_action[0][0]

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if not self.is_act_continuous:
                if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                    raise Exception("The input action dimension for player {} should be {}, not {}".format(
                        i, self.joint_action_space[i][0].n, len(joint_action[i][0])))
            else:
                if not isinstance(joint_action[i][0], np.ndarray):
                    raise Exception("For continuous action, the input of player {} should be numpy.ndarray".format(i))
                if joint_action[i][0].shape != self.joint_action_space[i][0].shape:
                    raise Exception("The input action dimension for player {} should be {}, not {}".format(
                        i, self.joint_action_space[i][0].shape, joint_action[i][0].shape))

    def get_next_state(self, action):
        observation, reward, done, info = self.env_core.step(action)
        obs_list = observation.tolist()
        return obs_list, reward, done, info

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
        if self.is_act_continuous:
            action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
        else:
            action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
        return action_space

    def check_win(self):
        return '0'

    def reset(self):
        observation = self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        # self.current_state = observation
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        obs_list = observation.reshape(-1).tolist()
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
            all_obs_space[i] = (m)
        return all_obs_space

    def get_vector_observation(self, current_state, player_id, info_before):
        return self.current_state[player_id]

    def set_seed(self, seed=None):
        self.env_core.seed(seed)

    def get_all_observes(self):
        all_observes = []
        for i in range(len(self.current_state)):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)
        return all_observes
