import numpy as np
from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
env = make("classic_MountainCar-v0")


class classic_MountainCar_v0(BaseWrapper):
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

        next_state, reward, done, _, _ = self.env.step(action)

        if train:
            # reward shapping
            reward = get_reward(next_state[0]["obs"])
            reward = np.array(reward)
            return next_state, reward, done, _, _

        else:
            reward = np.array(reward)
            return next_state, reward, done, _, _

    def reset(self):
        state = self.env.reset()
        return state

    def close(self):
        pass

    def set_seed(self, seed):
        self.env.set_seed(seed)


def get_reward(s_):
    if s_[0] > -0.5:  ##如果位置向右，就给奖励。
        r = s_[0] + 0.5 - 1
        if s_[0] > 0.5:  ###到达目标，给10的奖励
            r = 300
    else:
        r = float(-1)  ##在谷底不得分
    return r






