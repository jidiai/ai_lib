from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
env = make("gridworld")


class gridworld(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

    def get_actionspace(self):
        return self.env.action_dim

    def get_observationspace(self):
        return int(self.env.input_dimension)

    def step(self, action, train=True):
        '''
        return: next_state, reward, done, _, _
        '''

        next_state, reward, done, _, info = self.env.step(action)
        reward = reward[0]
        return next_state, reward, done, _, info

    def reset(self):
        state = self.env.reset()
        return state

    def close(self):
        pass

    def set_seed(self, seed=None):
        pass

