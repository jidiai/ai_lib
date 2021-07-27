# -*- coding:utf-8  -*-
# Time  : 2021/4/7 下午3:46
# Author: Yahui Cui
import copy

import make_env
import multiagent
import numpy as np
import gym
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete
from utils import mutli_discrete_particle


class ParticleEnv(Game, VectorObservation):
    def __init__(self, conf):
        super(ParticleEnv, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                          conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.done = False
        self.dones = []
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        self.env_core = make_env.make_env(conf["game_name"].split("-")[1])
        self.new_action_spaces = self.load_action_space()
        self.joint_action_space = self.set_action_space()

        self.init_info = None
        self.current_state = self.env_core.reset()
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.n_return = [0] * self.n_player
        self.action_dim = self.get_action_dim()
        self.input_dimension = self.env_core.observation_space

    def reset(self):
        self.step_cnt = 0
        self.done = False
        obs_list = self.env_core.reset()
        self.current_state = obs_list
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def step(self, joint_action):
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        next_state, reward, self.dones, info_after = \
            self.env_core.step(joint_action_decode)
        self.current_state = next_state
        self.all_observes = self.get_all_observes()
        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        for done in self.dones:
            if done == True:
                self.done = True
                break

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_vector_observation(self, current_state, player_id, info_before):
        return current_state[player_id]

    def load_action_space(self):
        origin_action_spaces = self.env_core.action_space
        new_action_spaces = []
        for action_space in origin_action_spaces:
            if isinstance(action_space, multiagent.multi_discrete.MultiDiscrete):
                low = action_space.low
                high = action_space.high
                array_of_param_array = []
                for x, y in zip(low, high):
                    array_of_param_array.append([x, y])
                new_action_spaces.append(mutli_discrete_particle.MultiDiscreteParticle(array_of_param_array))
            elif isinstance(action_space, gym.spaces.Discrete):
                new_action_spaces.append(Discrete(action_space.n))

        return new_action_spaces

    def set_action_space(self):
        action_space = [[self.new_action_spaces[i]] for i in range(self.n_player)]
        return action_space

    def check_win(self):
        return self.won

    def get_action_dim(self):
        action_dim = 1
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space)):
            item = self.joint_action_space[i][0]
            if isinstance(item, Discrete):
                action_dim *= item.n
            elif isinstance(item, mutli_discrete_particle.MultiDiscreteParticle):
                for k in range(len(item.high)):
                    action_dim *= (item.high[k] - item.low[k] + 1)

        return action_dim

    def decode(self, joint_action):
        joint_action_decode = []
        for nested_action in joint_action:
            if not isinstance(nested_action[0], np.ndarray):
                nested_action[0] = np.array(nested_action[0])
            joint_action_decode.append(nested_action[0])

        return np.array(joint_action_decode, dtype=object)

    def set_n_return(self, reward):
        for i in range(self.n_player):
            self.n_return[i] += reward[i]

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            item = copy.deepcopy(self.current_state[i])
            if isinstance(item, np.ndarray):
                item = item.tolist()
            each = {"obs": item, "controlled_player_index": i}
            all_observes.append(each)
        return all_observes
