# -*- coding:utf-8  -*-
# Time  : 2021/9/15 上午11:19
# Author: Yahui Cui


from env.simulators.game import Game
from utils.discrete import Discrete


import os
import sys
from pathlib import Path

CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
smarts_path = os.path.join(CURRENT_PATH, "SMARTS")
sys.path.append(smarts_path)
print(sys.path)
import copy
import gym
import numpy as np
from smarts.core.agent import AgentSpec, Agent
from smarts.core.agent_interface import AgentInterface, AgentType


class SmartsJidi(Game):
    def __init__(self, conf):
        super(SmartsJidi, self).__init__(
            conf["n_player"],
            conf["is_obs_continuous"],
            conf["is_act_continuous"],
            conf["game_name"],
            conf["agent_nums"],
            conf["obs_type"],
        )

        self.max_step = int(conf["max_step"])
        self.AGENT_ID = "Agent-007"
        self.agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
            agent_builder=SimpleAgent,
        )

        self.agent_specs = {self.AGENT_ID: self.agent_spec}

        scenario_name = conf["scenario_name"]
        scenario_path = os.path.join(CURRENT_PATH, "SMARTS", "scenarios", scenario_name)
        self.env_core = gym.make(
            "smarts.env:hiway-v0",
            scenarios=[scenario_path],
            agent_specs=self.agent_specs,
        )

        self.dones = {"__all__": False}
        self.done = False
        self.player_id_map, self.player_id_reverses_map = self.get_player_id_map(
            sorted(self.agent_specs.keys())
        )
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = None

        self.init_info = None
        self.all_observes = None
        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player
        _ = self.reset()

    def reset(self):
        self.step_cnt = 0
        self.dones = {"__all__": False}
        self.done = False
        self.init_info = None
        obs_list = self.env_core.reset()
        self.current_state = self.change_observation_keys(obs_list)
        self.all_observes = self.get_all_observevs()
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        all_observations, reward, self.dones, info_after = self.env_core.step(
            joint_action_decode
        )
        self.current_state = self.change_observation_keys(all_observations)
        self.all_observes = self.get_all_observevs()
        # print("debug all observes ", type(self.all_observes[0]["obs"]))
        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        # info_after = str(info_after)
        info_after = ""
        return self.all_observes, reward, done, info_before, info_after

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception(
                "Input joint action dimension should be {}, not {}".format(
                    self.n_player, len(joint_action)
                )
            )

        for i in range(self.n_player):
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                raise Exception(
                    "The input action dimension for player {} should be {}, not {}".format(
                        i, self.joint_action_space[i][0].n, len(joint_action[i][0])
                    )
                )

    def decode(self, joint_action):
        action_map = {
            0: "keep_lane",
            1: "slow_down",
            2: "change_lane_left",
            3: "change_lane_right",
        }
        agents_id_keys = sorted(self.agent_specs.keys())
        joint_action_decode = {}
        for act_id, nested_action in enumerate(joint_action):
            # print("debug nested_action ", nested_action)
            key = agents_id_keys[act_id]
            action_to_send = action_map[nested_action[0].index(1)]
            joint_action_decode[key] = action_to_send

        return joint_action_decode

    def step_before_info(self, info=""):
        return info

    def change_observation_keys(self, current_state):
        new_current_state = {}
        for key, state in current_state.items():
            changed_key = self.player_id_map[key]
            new_current_state[changed_key] = state

        return new_current_state

    def get_all_observevs(self):
        all_observes = []
        for i in range(self.n_player):
            if i not in self.current_state.keys():
                each = None
            else:
                each = copy.deepcopy(self.current_state[i])
            each = {"obs": each, "controlled_player_index": i}
            all_observes.append(each)
        return all_observes

    def get_player_id_map(self, player_keys):
        player_id_map = {}
        player_id_reverse_map = {}
        for i, key in enumerate(player_keys):
            player_id_map[key] = i
            player_id_reverse_map[i] = key
        return player_id_map, player_id_reverse_map

    def set_n_return(self, reward):
        for key in self.agent_specs.keys():
            if key in reward:
                changed_index = self.player_id_map[key]
                self.n_return[changed_index] += reward[key]

    def set_action_space(self):
        action_space = [[Discrete(4)] for _ in range(self.n_player)]
        return action_space

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        if self.dones["__all__"]:
            self.done = True

        if self.done:
            self.env_core.close()

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def check_win(self):
        return ""


class SimpleAgent(Agent):
    def act(self, obs):
        action_map = {
            0: "keep_lane",
            1: "slow_down",
            2: "change_lane_left",
            3: "change_lane_right",
        }
        action_id = np.random.randint(0, 4)
        return action_map[action_id]
