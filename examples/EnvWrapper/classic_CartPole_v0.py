import numpy as np
from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
env = make("classic_CartPole-v0")

class Cartpole_v0(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

    def get_actionspace(self):
        return self.env.action_dim

    def get_observationspace(self):
        return self.env.input_dimension.shape[0]

    def step(self, action, train=True):
        '''
        return: next_state, reward, done, _, _
        '''
        action = action_wrapper([action])
        next_state, reward, done, _, _ = self.env.step(action)

        reward = np.array(reward)
        next_state = np.array(next_state)
        return next_state, reward, done, _, _

    def reset(self):
        state = self.env.reset()
        return np.array(state)

    def close(self):
        pass

    def set_seed(self, seed):
        self.env.set_seed(seed)

def action_wrapper(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    for a in range(env.n_player):
        action_a = joint_action[a]["action"]
        each = [0] * env.action_dim
        each[action_a] = 1
        action_one_hot = [[each]]
        joint_action_.append([action_one_hot[0][0]])
    return joint_action_



