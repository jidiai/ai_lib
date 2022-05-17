import random
import os
import sys
from pathlib import Path

CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
olympics_path = os.path.join(CURRENT_PATH)
sys.path.append(olympics_path)

from olympics_engine.generator import create_scenario
from olympics_engine.scenario.wrestling import *

from utils.box import Box
from env.simulators.game import Game

import numpy as np

class OlympicsWrestling(Game):
    def __init__(self, conf, seed=None):
        super(OlympicsWrestling, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = seed
        self.set_seed()

        Gamemap = create_scenario("wrestling")
        self.env_core = wrestling(Gamemap)
        self.max_step = int(conf['max_step'])
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space

        self.step_cnt = 0
        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player

        _ = self.reset()

        self.board_width = self.env_core.view_setting['width']+2*self.env_core.view_setting['edge']
        self.board_height = self.env_core.view_setting['height']+2*self.env_core.view_setting['edge']

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        if not seed:        #use previous seed when no new seed input
            seed = self.seed
        else:               #update env global seed
            self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        init_obs = self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        self.won = {}
        self.n_return = [0]*self.n_player

        self.current_state = init_obs
        self.all_observes = self.get_all_observes()

        return self.all_observes


    def step(self, joint_action):
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        all_observations, reward, done, info_after = self.env_core.step(joint_action_decode)
        info_after = ''
        self.current_state = all_observations
        self.all_observes = self.get_all_observes()

        self.step_cnt += 1
        self.done = done
        if self.done:
            self.set_n_return()

        return self.all_observes, reward, self.done, info_before, info_after



    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:          #check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

    def step_before_info(self, info=''):
        return info

    def decode(self, joint_action):
        joint_action_decode = []
        for act_id, nested_action in enumerate(joint_action):
            temp_action = [0, 0]
            temp_action[0] = nested_action[0][0]
            temp_action[1] = nested_action[1][0]
            joint_action_decode.append(temp_action)

        return joint_action_decode

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)

        return all_observes

    def set_action_space(self):
        return [[Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))] for _ in range(self.n_player)]

    def get_reward(self, reward):
        return [reward]

    def is_terminal(self):
        return self.env_core.is_terminal()

    def set_n_return(self):

        if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
            self.n_return = [0,1]
        elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
            self.n_return = [1,0]
        elif self.env_core.agent_list[0].finished and self.env_core.agent_list[1].finished:
            self.n_return = [0,0]
        else:
            self.n_return = [0, 0]


    def check_win(self):

        if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
            return '1'
        elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
            return '0'
        else:
            return '-1'


    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]




