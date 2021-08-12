###  Based on code from Yahui Cui  ####

import copy
import numpy as np
from gym.utils import seeding
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete
from env.magent_base.battle_base import parallel_env


class Battle(Game, DictObservation):
    def __init__(self, conf):
        super(Battle, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                     conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.seed = None
        self.done = False
        self.dones = {}
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])
        self.map_size = None

        if "map_size" in conf.keys():
            self.map_size = int(conf["map_size"])

        self.env_core = parallel_env(map_size = self.map_size, max_cycles=500)

        self.player_id_map, self.player_id_reverses_map = self.get_player_id_map(self.env_core.action_spaces.keys())
        self.new_action_spaces = self.load_action_space()
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = self.env_core.observation_spaces

        self.init_info = None
        self.all_observes = None
        self.won = {}
        self.n_return = [0] * self.n_player
        _ = self.reset()

    def reset(self):
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        obs_list = self.env_core.reset()
        self.current_state = self.change_observation_keys(obs_list)
        self.all_observes = self.get_all_observevs()
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def step(self, joint_action):
        """
        joint_action: one hot action
        5       reward for killing an opponent
        -0.005  reward every step (step_reward option)
        -0.1    reward for attacking (attack_penalty option)
        0.2     reward for attacking an opponent (attack_opponent_reward option)
        -0.1    reward for dying (dead_penalty option)
        """
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        all_observations, reward, self.dones, info_after = \
            self.env_core.step(joint_action_decode)

        info_after = ''
        self.current_state = self.change_observation_keys(all_observations)
        self.all_observes = self.get_all_observevs()
        self.set_n_return(reward)
        self.step_cnt += 1
        return self.all_observes, reward, self.dones, info_before, info_after

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):

        if self.step_cnt == 0:   #initial state
            finished = False
        elif self.step_cnt >= self.max_step:    #excess maximun steps
            finished = True
        else:
            finished = all(self.dones.values())   #when either team is eliminated

        return finished

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_dict_observation(self, current_state, player_id, info_before):
        return current_state[player_id]

    def load_action_space(self):
        origin_action_spaces = self.env_core.action_spaces
        new_action_spaces = {}
        for key, action_space in origin_action_spaces.items():
            changed_key = self.player_id_map[key]
            new_action_spaces[changed_key] = Discrete(action_space.n)

        return new_action_spaces

    def set_action_space(self):
        action_space = [[self.new_action_spaces[i]] for i in range(self.n_player)]
        return action_space


    def check_win(self):
        left_sum = sum(self.n_return[:self.agent_nums[0]])
        right_sum = sum(self.n_return[self.agent_nums[0]:])
        if left_sum > right_sum:
            return '0'
        elif left_sum < right_sum:
            return '1'
        else:
            return '-1'

    def decode(self, joint_action):
        #print('joint action in base = ', joint_action)
        joint_action_decode = {}
        for act_id, nested_action in enumerate(joint_action):
            # print("debug nested_action ", nested_action)
            key = self.player_id_reverses_map[act_id]
            joint_action_decode[key] = nested_action.index(1)
            # joint_action_decode.append(nested_action[0])

        # return np.array(joint_action_decode, dtype=object)
        return joint_action_decode

    def set_n_return(self, reward):
        for key in self.env_core.action_spaces.keys():
            if key in reward:
                changed_index = self.player_id_map[key]
                self.n_return[changed_index] += reward[key]
        # for i in range(self.n_player):
        #     self.n_return[i] += reward[i]

    def change_observation_keys(self, current_state):
        new_current_state = {}
        for key, state in current_state.items():
            changed_key = self.player_id_map[key]
            new_current_state[changed_key] = state

        return new_current_state

    def get_player_id_map(self, player_keys):
        player_id_map = {}
        player_id_reverse_map = {}
        for i, key in enumerate(player_keys):
            player_id_map[key] = i
            player_id_reverse_map[i] = key
        return player_id_map, player_id_reverse_map

    def create_seed(self):
        seed = seeding.create_seed(None, max_bytes=4)
        return seed

    def set_seed(self, seed=None):
        self.env_core.seed(seed)
        self.seed = seed

    def get_all_observevs(self):
        all_observes = []
        for i in range(self.n_player):
            if i not in self.current_state.keys():
                each = None
            else:
                each = copy.deepcopy(self.current_state[i])
                if isinstance(each, np.ndarray):
                    each = each.tolist()
            each = {"obs": each, "controlled_player_index": i}
            all_observes.append(each)
        return all_observes
