# -*- coding:utf-8  -*-
# Time  : 2021/10/20 下午5:14
# Author: Yahui Cui


import gym
from env.simulators.game import Game


class GymRobotics(Game):
    def __init__(self, conf):
        super(GymRobotics, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                  conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.env_core = gym.make(self.game_name)

        self.max_step = int(conf["max_step"])
        self.done = False
        self.reach_goal = False

        self.current_state = self.env_core.reset()
        self.all_observes = self.get_all_observes()
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.input_dimension = self.env_core.observation_space

        self.init_info = None
        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player

    def reset(self):
        self.done = False
        self.reach_goal = False
        self.current_state = self.env_core.reset()
        self.all_observes = self.get_all_observes()
        self.init_info = None
        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player

    def step(self, joint_action):
        info_before = ''
        action = self.decode(joint_action)
        observation, reward, self.done, info = self.env_core.step(action)
        if info['is_success']:
            self.reach_goal = True
            self.done = True
        self.current_state = observation
        self.all_observes = self.get_all_observes()
        reward = self.get_reward(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        if done:
            self.set_final_n_return()
        info_after = info
        return self.all_observes, reward, done, info_before, info_after

    def get_all_observes(self):
        all_observes = []
        each = {"obs": self.current_state, "controlled_player_index": 0, "task_name": self.game_name}
        all_observes.append(each)
        return all_observes

    def set_action_space(self):
        action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
        return action_space

    def get_reward(self, reward):
        r = [0] * self.n_player
        # print("reward is ", reward)
        for i in range(self.n_player):
            r[i] = reward
            self.n_return[i] += r[i]

        return r

    def set_final_n_return(self):
        if self.reach_goal:
            for i in range(self.n_player):
                self.n_return[i] += self.max_step
        else:
            for i in range(self.n_player):
                self.n_return[i] = 0

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def decode(self, joint_action):
        return joint_action[0][0]

    def check_win(self):
        return ''
