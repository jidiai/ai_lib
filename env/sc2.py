# -*- coding:utf-8  -*-
# Time  : 2021/9/24 下午2:42
# Author: Yahui Cui
import copy
import sys
import numpy as np

from env.simulators.game import Game
from utils.discrete_sc2 import Discrete_SC2
from pysc2.env import sc2_env
from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)


class SC2(Game):
    def __init__(self, conf):
        super(SC2, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                         conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.players = [sc2_env.Agent(sc2_env.Race[agent_type]) for agent_type in conf["agent_type"]]

        self.env_core = sc2_env.SC2Env(map_name=conf["map_name"], players=self.players,
                             agent_interface_format=sc2_env.AgentInterfaceFormat(
                                 feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64)), step_mul=16,
                             game_steps_per_episode=200 * 16)

        self.max_step = int(conf["max_step"])
        self.dones = False
        self.done = False
        timesteps = self.env_core.reset()
        self.current_state = timesteps
        self.all_observes = self.get_all_observevs()
        self.joint_action_space = self.set_action_space(timesteps)
        self.action_dim = self.joint_action_space
        self.input_dimension = None

        self.init_info = None
        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player

    def reset(self):
        timesteps = self.env_core.reset()
        self.current_state = timesteps
        self.all_observes = self.get_all_observevs()
        self.joint_action_space = self.set_action_space(timesteps)
        self.action_dim = self.joint_action_space
        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player

    def step(self, joint_action):
        info_before = ''
        joint_action_decode = self.decode(joint_action)
        timesteps = self.env_core.step(joint_action_decode)
        self.current_state = timesteps
        self.all_observes = self.get_all_observevs()
        reward = self.set_n_return()
        done = self.is_terminal()
        self.joint_action_space = self.set_action_space(timesteps)
        self.step_cnt += 1
        info_after = ''
        return self.all_observes, reward, done, info_before, info_after

    def set_action_space(self, timesteps):
        new_joint_action_space = []
        for timestep, agent_spec in zip(timesteps, self.env_core.action_spec()):
            new_joint_action_space.append([Discrete_SC2(timestep.observation.available_actions, agent_spec)])
        return new_joint_action_space

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def decode(self, joint_action):
        joint_action_decode = []
        for act in joint_action:
            joint_action_decode.append(act[0])
        return joint_action_decode

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if joint_action[i][0].function not in self.joint_action_space[i][0].available_actions:
                raise Exception("The input action dimension for player {} should be {}, does not have {}".format(
                    i, self.joint_action_space[i][0].available_actions, joint_action[i][0].function))

    def get_all_observevs(self):
        all_observes = []
        for i in range(self.n_player):
            each = copy.deepcopy(self.current_state[i])
            each = {"obs": each, "controlled_player_index": i}
            all_observes.append(each)
        return all_observes

    def set_n_return(self):
        reward = []
        for idx, obs in enumerate(self.current_state):
            self.n_return[idx] += obs.reward
            reward.append(obs.reward)
        return reward

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        for obs in self.current_state:
            if obs.last():
                self.done = True

        return self.done

    def check_win(self):
        if len(self.n_return) == 1:
            return ''
        else:
            all_equal = True
            for i in range(1, len(self.n_return)):
                if self.n_return[i-1] != self.n_return[i]:
                    all_equal = False
                    break
            if all_equal:
                return -1
            return np.argmax(self.n_return)
