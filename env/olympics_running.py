import time
import math
import random
import numpy as np
import os
import sys
from pathlib import Path

CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
olympics_path = os.path.join(CURRENT_PATH)
sys.path.append(olympics_path)

object_path = '/olympics'
sys.path.append(os.path.join(olympics_path+object_path))
print('sys path = ', sys.path)

from olympics_engine.core import OlympicsBase
from olympics_engine.generator import create_scenario
from olympics_engine.scenario.running_competition import *

from env.simulators.game import Game
from utils.box import Box


import argparse
import json
def store(record, name):
    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson


class OlympicsRunning(Game):
    def __init__(self, conf, seed=None):
        super(OlympicsRunning, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                         conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.seed = seed
        self.set_seed()

        #choose a map randomly
        self.num_map = conf['map_num']
        map_index_seq = list(range(1, conf['map_num']+1))
        rand_map_idx = random.choice(map_index_seq)
        Gamemap = create_scenario("running-competition")

        self.env_core = Running_competition(meta_map = Gamemap, map_id = rand_map_idx)
        self.env_core.VIEW_BACK = -self.env_core.map['agents'][0].r/self.env_core.map['agents'][0].visibility
        self.env_core.VIEW_ITSELF = True
        self.env_core.obs_boundary_init = list()
        self.env_core.obs_boundary = self.env_core.obs_boundary_init

        for index, item in enumerate(self.env_core.map["agents"]):
            position = item.position_init
            r = item.r
            if item.type == 'agent':
                visibility = item.visibility
                boundary = self.env_core.get_obs_boundaray(position, r, visibility)
                self.env_core.obs_boundary_init.append(boundary)
            else:
                self.env_core.obs_boundary_init.append(None)


        self.max_step = int(conf['max_step'])
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space

        self.env_core.map_num = rand_map_idx

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

    def specify_a_map(self, num):
        assert num <= self.num_map, print('the num is larger than the total number of map')
        Gamemap = create_scenario("map"+str(num))
        self.env_core = Running_competition(map_id = num)
        _ = self.reset()
        self.env_core.map_num = num

    def reset(self, shuffle_map=False):
        #self.set_seed()

        if shuffle_map:   #if shuffle the map, randomly sample a map again
            map_index_seq = list(range(1, self.num_map + 1))
            rand_map_idx = random.choice(map_index_seq)
            Gamemap = create_scenario("map"+str(rand_map_idx))
            self.env_core = Running_competition(map_id = rand_map_idx)
            self.env_core.map_num = rand_map_idx

        self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player

        self.current_state = self.env_core.get_obs()        #[2,100,100] list
        self.all_observes = self.get_all_observes()         #wrapped obs with player index

        return self.all_observes


    def step(self, joint_action):
        #joint_action: should be [  [[0,0,1], [0,1,0]], [[1,0,0], [0,0,1]]  ] or [[array(), array()], [array(), array()]]
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
        return [[Box(-100,200, shape=(1,)), Box(-30, 30, shape=(1,))] for _ in range(self.n_player)]

    def get_reward(self, reward):
        return [reward]

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True
        for agent_idx in range(self.n_player):
            if self.env_core.agent_list[agent_idx].finished:
                return True

        return False

    def set_n_return(self):

        if self.env_core.agent_list[0].finished and not(self.env_core.agent_list[1].finished):
            self.n_return = [1,0]
        elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
            self.n_return = [0,1]
        elif self.env_core.agent_list[0].finished and self.env_core.agent_list[1].finished:
            self.n_return = [1,1]
        else:
            self.n_return = [0,0]

    def check_win(self):
        if self.env_core.agent_list[0].finished and not(self.env_core.agent_list[1].finished):
            return '0'
        elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
            return '1'
        else:
            return '-1'


    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]







