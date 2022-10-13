# -*- coding:utf-8  -*-
# Time  : 2021/12/28 下午4:43
# Author: Yahui Cui


import copy
import numpy as np

from gym.utils import seeding
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete
from utils.box import Box

from pettingzoo.mpe import simple_v2
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_crypto_v2
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_reference_v2
from pettingzoo.mpe import simple_speaker_listener_v3
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.mpe import simple_world_comm_v2


class MPE_Jidi(Game, DictObservation):
    def __init__(self, conf):
        super(MPE_Jidi, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                     conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = None
        self.done = False
        self.dones = {}
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        env_name = conf["game_name"].split("-")[1]
        action_continues = self.is_act_continuous
        self.env_core = None
        if env_name == "simple":
            self.env_core = simple_v2.parallel_env(max_cycles=25, continuous_actions=action_continues)
        elif env_name == "simple_adversary":
            self.env_core = simple_adversary_v2.parallel_env(N=2, max_cycles=25, continuous_actions=action_continues)
        elif env_name == "simple_crypto":
            self.env_core = simple_crypto_v2.parallel_env(max_cycles=25, continuous_actions=action_continues)
        elif env_name == "simple_push":
            self.env_core = simple_push_v2.parallel_env(max_cycles=25, continuous_actions=action_continues)
        elif env_name == "simple_reference":
            self.env_core = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=25,
                                                             continuous_actions=action_continues)
        elif env_name == "simple_speaker_listener":
            self.env_core = simple_speaker_listener_v3.parallel_env(max_cycles=25, continuous_actions=action_continues)
        elif env_name == "simple_spread":
            self.env_core = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25,
                                                          continuous_actions=action_continues)
        elif env_name == "simple_tag":
            self.env_core = simple_tag_v2.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25,
                                                       continuous_actions=action_continues)
        elif env_name == "simple_world_comm":
            self.env_core = simple_world_comm_v2.parallel_env(num_good=2, num_adversaries=4, num_obstacles=1,
                                                              num_food=2, max_cycles=25, num_forests=2,
                                                              continuous_actions=action_continues)

        if self.env_core is None:
            raise Exception("MPE_Jidi env_core is None!")

        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player
        self.step_cnt = 0
        self.done = False
        self.player_id_map, self.player_id_reverse_map = self.get_player_id_map(self.env_core.action_spaces.keys())

        # set up action spaces
        self.new_action_spaces = self.load_action_space()
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = self.env_core.observation_spaces

        # set up first all_observes
        obs = self.env_core.reset()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.dones = {agent: False for agent in self.env_core.possible_agents}

    def reset(self):
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        obs = self.env_core.reset()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.n_return = [0] * self.n_player
        self.dones = {agent: False for agent in self.env_core.possible_agents}
        return self.all_observes

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        obs, reward, self.dones, info_after = self.env_core.step(joint_action_decode)
        info_after = ''
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        # print("debug all observes ", type(self.all_observes[0]["obs"]))
        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}.".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            player_name = self.player_id_reverse_map[i]
            if joint_action[i] is None or joint_action[i][0] is None:
                continue
            if not self.is_act_continuous:
                if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                    raise Exception("The input action dimension for player {}, {} should be {}, not {}.".format(
                        i, player_name, self.joint_action_space[i][0].n, len(joint_action[i][0])))
                if not (1 in joint_action[i][0]):
                    raise Exception("The input should be an one-hot vector!")
            else:
                if np.array(joint_action[i][0]).shape != self.joint_action_space[i][0].shape:
                    raise Exception("The input action dimension for player {}, {} should be {}, not {}.".format(
                        i, player_name, self.joint_action_space[i][0].shape, np.array(joint_action[i][0]).shape))

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        if not self.env_core.agents:
            self.done = True

        if all(self.dones.values()):
            self.done = True

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def load_action_space(self):
        origin_action_spaces = self.env_core.action_spaces
        new_action_spaces = {}
        for key, action_space in origin_action_spaces.items():
            changed_key = self.player_id_map[key]
            if not self.is_act_continuous:
                new_action_spaces[changed_key] = Discrete(action_space.n)
            else:
                new_action_spaces[changed_key] = Box(action_space.low, action_space.high,
                                                     action_space.shape, np.float32)

        return new_action_spaces

    def set_action_space(self):
        action_space = [[self.new_action_spaces[i]] for i in range(self.n_player)]
        return action_space

    def check_win(self):
        if len(self.agent_nums) == 1:
            return self.won

        left = sum(self.n_return[0:self.agent_nums[0]])
        right = sum(self.n_return[self.agent_nums[0]:])

        if left > right:
            return "0"
        elif left > right:
            return "1"
        else:
            return "-1"

    def decode(self, joint_action):
        joint_action_decode = {}
        for act_id, nested_action in enumerate(joint_action):
            # print("debug nested_action ", nested_action)
            key = self.player_id_reverse_map[act_id]
            if nested_action is None or nested_action[0] is None:
                continue
            if not self.is_act_continuous:
                if isinstance(nested_action[0], np.ndarray):
                    nested_action[0] = nested_action[0].tolist()
                joint_action_decode[key] = nested_action[0].index(1)
            else:
                joint_action_decode[key] = nested_action[0]
            # joint_action_decode.append(nested_action[0])

        # return np.array(joint_action_decode, dtype=object)
        return joint_action_decode

    def set_n_return(self, reward):
        for player_key, player_reward in reward.items():
            player_id = self.player_id_map[player_key]
            self.n_return[player_id] += player_reward

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

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            player_name = self.player_id_reverse_map[i]
            each_obs = copy.deepcopy(self.current_state[player_name])
            each = {"obs": each_obs, "controlled_player_index": i, "controlled_player_name": player_name}
            all_observes.append(each)
        return all_observes

    def all_equals(self, list_to_compare):
        return len(set(list_to_compare)) == 1
