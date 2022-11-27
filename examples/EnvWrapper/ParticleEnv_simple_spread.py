from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
import os
import numpy as np

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
print("+", base_dir)
from env.chooseenv import make

env = make("ParticleEnv-simple_spread")


class ParticleEnv_simple_spread(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

    def get_actionspace(self):
        action_dim = [
            self.env.get_single_action_space(id)[0].n for id in range(self.env.n_player)
        ]
        print("##", action_dim)
        return action_dim

    def get_observationspace(self):
        obs_dim = [
            self.env.current_state[id].shape[0] for id in range(self.env.n_player)
        ]
        print("##", obs_dim)
        return obs_dim

    def step(self, action, train=True):
        action = [[np.array(a, dtype=np.float)] for a in action]
        next_state, reward, done, _, _ = self.env.step(action)
        reward = np.array(reward, dtype=np.float)
        next_state = [next_state[id] for id in range(self.env.n_player)]
        done = False  # 不能因为episode到头就给True，会影响Q值
        return next_state, reward, done, _, _

    def reset(self):
        state = self.env.reset()
        obs = [state[id] for id in range(self.env.n_player)]
        return obs

    def close(self):
        pass

    def set_seed(self, seed):
        pass


if __name__ == "__main__":
    env = ParticleEnv_simple_spread()
    a, b, c, d, e = env.step([[1, 2, 3, 4, 5]] * 3)
