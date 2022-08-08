# -*- coding:utf-8  -*-
# Time  : 2022/7/25 下午4:19
# Author: Yahui Cui
import copy
import random
import numpy as np

from env.simulators.game import Game
from env.revive.refrigerator import DoorOpen, Simulator
from utils.box import Box


class Refrigerator(Game):
    def __init__(self, conf, seed=0):
        super(Refrigerator, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                           conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.max_step = int(conf["max_step"])
        self.done = False
        self.seed = None
        self.set_seed(seed)
        self.won = {}
        self.n_return = [0] * self.n_player

        self.step_cnt = 0
        self.open_interval = 200
        self.open_door = False
        self.door_open_after_step = self.open_interval + 1
        self.init_temperature = 10
        self.target_temperature = -2
        self.sampling_time = 1
        self.door_open_agent = DoorOpen(door_open_time=20)
        self.cool_simulator = Simulator(self.init_temperature)
        self.cool_simulator.reset(init_temperature=self.init_temperature)
        self.traj = []

        self.joint_action_space = self.set_action_space()
        self.current_state = self.cool_simulator.get_temperature()
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after(False)

    def reset(self):
        self.won = {}
        self.n_return = [0] * self.n_player
        self.step_cnt = 0
        self.open_interval = 200
        self.open_door = False
        self.door_open_after_step = self.open_interval + 1
        self.init_temperature = 10
        self.target_temperature = -2
        self.sampling_time = 1
        self.door_open_agent = DoorOpen(door_open_time=20)
        self.cool_simulator = Simulator(self.init_temperature)
        self.cool_simulator.reset(init_temperature=self.init_temperature)
        self.traj = []

        self.current_state = self.cool_simulator.get_temperature()
        self.all_observes = self.get_all_observes()
        self.init_info = self.get_info_after(False)
        return self.all_observes

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        if self.step_cnt % self.open_interval == 0:
            if random.random() < 0.5:
                self.open_door = True
            self.door_open_agent.reset()
            self.door_open_after_step = random.randint(0, self.open_interval - self.door_open_agent.door_open_time)
        action = joint_action[0][0]
        if self.open_door and self.step_cnt % self.open_interval >= self.door_open_after_step:
            door_open = self.door_open_agent.act()
            self.cool_simulator.update(power=action, dt=self.sampling_time, door_open=door_open)
        else:
            door_open = False
            self.cool_simulator.update(power=action, dt=self.sampling_time, door_open=door_open)
        self.current_state = self.cool_simulator.get_temperature()
        self.all_observes = self.get_all_observes()
        self.traj.append(copy.deepcopy(self.current_state))
        reward = -(abs(self.current_state - self.init_temperature))
        self.step_cnt += 1
        done = self.is_terminal()
        if done:
            self.set_n_return()
        info_after = self.get_info_after(door_open)

        return self.all_observes, reward, done, '', info_after

    def is_valid_action(self, joint_action):

        if np.isscalar(joint_action):
            raise Exception("Input joint action dimension should be (1,)")

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be (1,)")

        if np.isscalar(joint_action[0]):
            raise Exception("Input joint action dimension should be (1,)")

        if len(joint_action[0]) != 1:
            raise Exception("Input joint action dimension should be (1,)")

        if isinstance(joint_action[0][0], np.ndarray):
            joint_action[0][0] = joint_action[0][0][0]

        if not np.isscalar(joint_action[0][0]):
            raise Exception("Value in the action should be a scalar")

        if joint_action[0][0] < 0 or joint_action[0][0] > 10:
            raise Exception("Value of action should between 0 and 10")

    def set_action_space(self):
        return [[Box(low=0, high=10, shape=(1,))]]

    def get_all_observes(self):
        return [{"obs": copy.deepcopy(self.current_state), "controlled_player_index": 0}]

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done

    def set_seed(self, seed):
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)

    def get_info_after(self, door_open):
        return {"temperature": copy.deepcopy(self.current_state), "controlled_player_index": 0, "door_open": door_open}

    def set_n_return(self):
        self.n_return[0] = -np.mean(np.abs(np.array(self.traj) - self.target_temperature))

    def check_win(self):
        return self.won

