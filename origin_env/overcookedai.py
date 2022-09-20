# -*- coding:utf-8  -*-
# Time  : 2021/6/4 上午9:45
# Author: Yahui Cui
import copy

from env.simulators.game import Game
from env.obs_interfaces.observation import *
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from utils.discrete import Discrete

import gym
import numpy as np


class OvercookedAI(Game, DictObservation):
    def __init__(self, conf):
        super(OvercookedAI, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                           conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        self.base_mdp = OvercookedGridworld.from_layout_name(conf["game_name"].split("-")[1])
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        obs_list = self.env.reset()
        self.current_state = [obs_list.to_dict() if obs_list is not None else obs_list for _ in range(self.n_player)]
        self.all_observes = self.get_all_observes()

        self.joint_action_space = self.set_action_space()

        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player
        self.action_dim = self.get_action_dim()
        # self.observation_space = self._setup_observation_space()
        self.input_dimension = None

    def reset(self):
        self.step_cnt = 0
        self.done = False
        obs_list = self.env.reset()
        self.current_state = [obs_list.to_dict() if obs_list is not None else obs_list for _ in range(self.n_player)]
        self.all_observes = self.get_all_observes()
        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def step(self, joint_action):
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        next_state, reward, self.done, info_after = self.env.step(joint_action_decode)
        next_state = next_state.to_dict()
        self.current_state = [next_state for _ in range(self.n_player)]
        self.all_observes = self.get_all_observes()
        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        info_after = self.parse_info_after(info_after)
        return self.all_observes, reward, done, info_before, info_after

    def step_before_info(self, info=''):
        return info

    def parse_info_after(self, info_after):
        if 'episode' in info_after:
            episode = info_after['episode']
            for out_key, out_value in episode.items():
                if isinstance(out_value, dict):
                    for key, value in out_value.items():
                        if isinstance(value, np.ndarray):
                            info_after['episode'][out_key][key] = value.tolist()
                elif isinstance(out_value, np.ndarray):
                    info_after['episode'][out_key] = out_value.tolist()
                elif isinstance(out_value, np.int64):
                    info_after['episode'][out_key] = int(out_value)

        return info_after

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_dict_observation(self, current_state, player_id, info_before):
        return current_state

    def set_action_space(self):
        origin_action_space = self.action_space
        new_action_spaces = []
        for _ in range(self.n_player):
            new_action_spaces.append([Discrete(origin_action_space.n)])

        return new_action_spaces

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

        return action_dim

    def decode(self, joint_action):
        joint_action_decode = []
        joint_action_decode_tmp = []
        for nested_action in joint_action:
            if not isinstance(nested_action[0], np.ndarray):
                nested_action[0] = np.array(nested_action[0])
            joint_action_decode_tmp.append(nested_action[0].tolist().index(1))

        for action_id in joint_action_decode_tmp:
            joint_action_decode.append(Action.INDEX_TO_ACTION[action_id])

        return joint_action_decode

    def set_n_return(self, reward):
        for i in range(self.n_player):
            self.n_return[i] += reward

    def set_seed(self, seed=0):
        np.random.seed(seed)

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = copy.deepcopy(self.current_state[i])
            if each:
                each["controlled_player_index"] = i
            all_observes.append(each)
        return all_observes
