import numpy as np
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from env.ccgame import CCGame
import os
import json


def action_wrapper(action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''

    single_env_action = list()
    each = [0] * 2
    each[action] = 1
    single_env_action.append([each])

    return single_env_action


def make_env_wrapper(game_name):
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env' + '\config.json')
    with open(file_path) as f:
        conf = json.load(f)[game_name]
    env = classic_CartPole_v0(conf)
    return env


class classic_CartPole_v0(CCGame):
    def __init__(self, config):
        super().__init__(config)

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        action = self.decode(joint_action)
        info_before = self.step_before_info()
        next_state, reward, self.done, info_after = self.get_next_state(action)
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()[0]
        reward = self.get_reward(reward)
        self.current_state = [next_state] * self.n_player
        self.step_cnt += 1
        done = self.is_terminal()
        return next_state, reward, done, info_before, info_after

    def reset(self):
        observation = self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        return observation


