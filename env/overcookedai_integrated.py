# -*- coding:utf-8  -*-

import copy
import random

from env.simulators.game import Game
from env.obs_interfaces.observation import *
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from utils.discrete import Discrete
# from gym.spaces.discrete import Discrete

import gym
import numpy as np

__all__ = ['OvercookedAI_Integrated']

class OvercookedAI_Integrated(Game, DictObservation):
    def __init__(self, conf):
        super(OvercookedAI_Integrated, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                           conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        self.game_pool = ['forced_coordination','forced_coordination',
                          'coordination_ring', 'coordination_ring',
                          'cramped_room', 'cramped_room']
        # random.shuffle(self.game_pool)
        self.agent_mapping = [[0,1],[1,0]]*3
        self.player2agent_mapping = None

        self.reset_map()
        self.env.display_states(self.env.state)


        self.action_space = Discrete(len(Action.ALL_ACTIONS))

        self.joint_action_space = self.set_action_space()

        self.init_info = {"game_pool": self.game_pool, "agent_mapping": self.agent_mapping}
        self.won = {}
        self.n_return = [0] * self.n_player
        self.action_dim = self.get_action_dim()
        # self.observation_space = self._setup_observation_space()
        self.input_dimension = None

    def reset_map(self):

        map_name = self.game_pool.pop(0)
        base_mdp = OvercookedGridworld.from_layout_name(map_name)
        self.env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS)
        self.current_game = map_name
        self.env.reset()
        global_state = self.env.state.to_dict()
        self.player_state = [global_state for _ in range(self.n_player)]            #share across agents

        #shuffle agent idx
        self.player2agent_mapping = self.agent_mapping.pop(0)
        print(f'Game {6 - len(self.game_pool)}: {map_name}, agent index: {self.player2agent_mapping}')
        # random.shuffle(self.player2agent_mapping)
        self.all_observes = self.get_all_observes(new_map=True)

    @property
    def last_game(self):
        return (len(self.game_pool)==0)

    def reset(self):
        self.step_cnt = 0
        self.done = False

        self.game_pool = ['forced_coordination','forced_coordination',
                          'coordination_ring', 'coordination_ring',
                          'cramped_room', 'cramped_room']
        self.agent_mapping = [[0,1],[1,0]]*3
        self.player2agent_mapping = None
        self.reset_map()

        self.init_info = {"game_pool": self.game_pool, "agent_mapping": self.agent_mapping}
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def step(self, joint_action):

        joint_action_decode = self.decode(joint_action)
        info_before = self.step_before_info(joint_action_decode)
        next_state, reward, map_done, info_after = self.env.step(joint_action_decode)
        if map_done and not self.last_game:
            self.reset_map()
            self.env.display_states(self.env.state)
            info_after['new_map'] = True

        else:
            next_state = next_state.to_dict()
            self.player_state = [next_state for _ in range(self.n_player)]
            # self.current_state = [next_state for _ in range(self.n_player)]
            self.all_observes = self.get_all_observes()
        if reward >0:
            print('reward', reward)
            raise NotImplementedError

        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        info_after = self.parse_info_after(info_after)
        return self.all_observes, reward, done, info_before, info_after

    def step_before_info(self, env_action):
        info = {
            "env_actions": env_action,
            "player2agent_mapping": self.player2agent_mapping
        }

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

        info_after['state'] = self.env.state.to_dict()
        info_after['player2agent_mapping'] = self.player2agent_mapping

        return info_after

    def is_terminal(self):
        if self.step_cnt >= self.max_step and self.last_game:
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

        #swap according to agent index
        joint_action_decode_tmp2 = [joint_action_decode_tmp[i] for i in self.player2agent_mapping]


        for action_id in joint_action_decode_tmp2:
            joint_action_decode.append(Action.INDEX_TO_ACTION[action_id])

        return joint_action_decode

    def set_n_return(self, reward):
        for i in range(self.n_player):
            self.n_return[i] += reward

    def set_seed(self, seed=0):
        np.random.seed(seed)

    def get_all_observes(self, new_map=False):
        all_observes = []
        for player_idx in range(self.n_player):
            mapped_agent_idx = self.player2agent_mapping[player_idx]
            each = copy.deepcopy(self.player_state[mapped_agent_idx])
            if each:
                each["controlled_player_index"] = mapped_agent_idx
                each['new_map'] = new_map
            all_observes.append(each)
        return all_observes
