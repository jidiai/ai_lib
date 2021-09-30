



import time
import math
import os
import sys
from pathlib import Path

CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
olympics_path = os.path.join(CURRENT_PATH)
sys.path.append(olympics_path)

from OlympicsEnv.olympics.scenario.running import *



def closest_point(l1, l2, point):

    A1 = l2[1] - l1[1]
    B1 = l1[0] - l2[0]
    C1 = (l2[1] - l1[1])*l1[0] + (l1[0] - l2[0])*l1[1]
    C2 = -B1 * point[0] + A1 * point[1]
    det = A1*A1 + B1*B1
    if det == 0:
        cx, cy = point
    else:
        cx = (A1*C1 - B1*C2)/det
        cy = (A1*C2 + B1*C1)/det

    return [cx, cy]

def distance_to_line(l1, l2, pos):
    closest_p = closest_point(l1, l2, pos)

    n = [pos[0] - closest_p[0], pos[1] - closest_p[1]]  # compute normal
    nn = n[0] ** 2 + n[1] ** 2
    nn_sqrt = math.sqrt(nn)
    cl1 = [l1[0] - pos[0], l1[1] - pos[1]]
    cl1_n = (cl1[0] * n[0] + cl1[1] * n[1]) / nn_sqrt

    return abs(cl1_n)






from env.simulators.game import Game
from utils.discrete import Discrete


class olympics_running(Game):
    def __init__(self, conf):

        super(olympics_running, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                         conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.env_core = Running()
        self.max_step = int(conf['max_step'])

        self.action_dim = [len(self.env_core.action_f), len(self.env_core.action_theta)]
        self.joint_action_space = self.set_action_space()       #[[Discrete(3), Discrete(3)], [Discrete(3), Discrete(3)]]

        self.step_cnt = 0

        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player

        _ = self.reset()

    def reset(self):

        self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player

        self.current_state = self.env_core.get_obs()        #[2,100,100] list
        self.all_observes = self.get_all_observes()         #wrapped obs with player index

        return self.all_observes

    def step(self, joint_action):       #joint_action: should be [  [[0,0,1], [0,1,0]], [[1,0,0], [0,0,1]]  ]
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        all_observations, reward, done, info_after = self.env_core.step(joint_action_decode)

            #[2,100,100] list
        info_after = ''
        self.current_state = all_observations
        self.all_observes = self.get_all_observes()

        self.step_cnt += 1

        self.done = self.is_terminal()

        if self.done:
            self.d1, self.d2 = self.compute_distance()
            self.set_n_return(self.d1, self.d2)

        return self.all_observes, reward, self.done, info_before, info_after


    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:          #check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if len(joint_action[i][0]) != self.joint_action_space[i][0].n:      #check the dimension of force
                raise Exception("The input action dimension of driving force for player {} should be {}, not {}".format(
                    i, self.joint_action_space[i][0].n, len(joint_action[i][0])))
            if len(joint_action[i][1]) != self.joint_action_space[i][1].n:      #check the dimension of theta
                raise Exception("The input action dimension of turing angle for player {} should be {}, not {}".format(
                    i, self.joint_action_space[i][1].n, len(joint_action[i][1])))

    def step_before_info(self, info=''):
        return info


    def set_action_space(self):
        action_space = [[Discrete(3), Discrete(3)] for _ in range(self.n_player)]
        return action_space


    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)

        return all_observes


    def decode(self, joint_action):     #one hot to integer action

        joint_action_decode = []
        for act_id, nested_action in enumerate(joint_action):
            temp_action = [0, 0]
            temp_action[0] = nested_action[0].index(1)
            temp_action[1] = nested_action[1].index(1)
            joint_action_decode.append(temp_action)

        return joint_action_decode

    def get_reward(self, reward):
        return [reward]

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.n_player):
            if self.env_core.agent_list[agent_idx].finished:
                return True

        return False

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]


    def compute_distance(self):
        for object_idx in range(len(self.env_core.map['objects'])):
            object = self.env_core.map['objects'][object_idx]
            if object.color == 'red':
                l1, l2 = object.init_pos
                distance1 = distance_to_line(l1, l2, self.env_core.agent_pos[0])
                distance2 = distance_to_line(l1, l2, self.env_core.agent_pos[1])

        return distance1, distance2

    def check_win(self):

        #find the cross first
        #for object_idx in range(len(self.env_core.map['objects'])):
        #    object = self.env_core.map['objects'][object_idx]
        #    if object.color == 'red':
        #        l1, l2 = object.init_pos
        #        distance1 = distance_to_line(l1, l2, self.env_core.agent_pos[0])
        #        distance2 = distance_to_line(l1, l2, self.env_core.agent_pos[1])
        distance1, distance2 = self.d1, self.d2

        if distance1 < distance2:
            return "0"
        elif distance1 > distance2:
            return "1"
        else:
            return "-1"


    def set_n_return(self, d1, d2):

        self.n_return[0] = -d1
        self.n_return[1] = -d2







