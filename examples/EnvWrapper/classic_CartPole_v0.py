import numpy as np
from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
env = make("classic_CartPole-v0")


class classic_CartPole_v0(BaseWrapper):
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
        reward = np.array(reward)
        return next_state, reward, done, _, _

    def reset(self):
        state = self.env.reset()
        return state

    def close(self):
        pass

    def set_seed(self, seed):
        self.env.set_seed(seed)

    def make_render(self):
        self.env.env_core.render()





