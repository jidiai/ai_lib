# -*- coding:utf-8  -*-
# Time  : 2021/11/3 上午10:05
# Author: Yahui Cui


import copy
import gym
from env.simulators.game import Game
from env.obs_interfaces.observation import *
from utils.discrete import Discrete


class ChineseChess(Game, DictObservation):
    def __init__(self, conf):
        super(ChineseChess, self).__init__(
            conf["n_player"],
            conf["is_obs_continuous"],
            conf["is_act_continuous"],
            conf["game_name"],
            conf["agent_nums"],
            conf["obs_type"],
        )
        self.done = False
        self.dones = {}
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        env_name = conf["game_name"]

        self.env_core_no_render = gym.make(env_name)

        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player
        self.step_cnt = 0
        self.done = False
        obs = self.env_core_no_render.reset()

        # set up action spaces
        self.new_action_spaces = self.load_action_space()
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = self.env_core_no_render.observation_space

        # set up first all_observes
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.current_player = self.env_core_no_render.current_player

    def reset(self):
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        obs = self.env_core_no_render.reset()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.n_return = [0] * self.n_player
        self.current_player = self.env_core_no_render.current_player
        return self.all_observes

    def step(self, joint_action):
        self.current_player = self.env_core_no_render.current_player
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        # Making an invalid move ends the game
        if joint_action_decode not in self.env_core_no_render.get_possible_actions():
            reward = -1
            self.done = True
            self.set_n_return(reward)
            return self.all_observes, reward, self.done, "", ""
        obs, reward, self.done, info_after = self.env_core_no_render.step(
            joint_action_decode
        )
        info_after = ""
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        # print("debug all observes ", type(self.all_observes[0]["obs"]))
        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception(
                "Input joint action dimension should be {}, not {}.".format(
                    self.n_player, len(joint_action)
                )
            )

        if (
            joint_action[self.current_player] is None
            or joint_action[self.current_player][0] is None
        ):
            raise Exception(
                "Action of current player is needed. Current player is {}".format(
                    self.current_player
                )
            )

        for i in range(self.n_player):
            if joint_action[i] is None or joint_action[i][0] is None:
                continue
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                raise Exception(
                    "The input action dimension for player {} should be {}, not {}.".format(
                        i, self.joint_action_space[i][0].n, len(joint_action[i][0])
                    )
                )

    def step_before_info(self, info=""):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def load_action_space(self):
        origin_action_space = self.env_core_no_render.action_space
        new_action_spaces = {}
        for player_id in range(self.n_player):
            new_action_spaces[player_id] = Discrete(origin_action_space.n)

        return new_action_spaces

    def set_action_space(self):
        action_space = [[self.new_action_spaces[i]] for i in range(self.n_player)]
        return action_space

    def check_win(self):
        if self.all_equals(self.n_return):
            return "-1"

        index = []
        max_n = max(self.n_return)
        for i in range(len(self.n_return)):
            if self.n_return[i] == max_n:
                index.append(i)

        if len(index) == 1:
            return str(index[0])
        else:
            return str(index)

    def decode(self, joint_action):
        if (
            joint_action[self.current_player] is None
            or joint_action[self.current_player][0] is None
        ):
            return None
        joint_action_decode = joint_action[self.current_player][0].index(1)
        return joint_action_decode

    def set_n_return(self, reward):

        self.n_return[self.current_player] += reward

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each_obs = {
                "observation": copy.deepcopy(self.current_state),
                "possible_actions": self.env_core_no_render.get_possible_actions(),
            }
            each = {"obs": each_obs, "controlled_player_index": i}
            all_observes.append(each)
        return all_observes

    def all_equals(self, list_to_compare):
        return len(set(list_to_compare)) == 1
