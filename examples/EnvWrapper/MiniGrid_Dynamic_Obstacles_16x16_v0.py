from examples.EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
import os
import numpy as np

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
print("+", base_dir)
from env.chooseenv import make

env = make('MiniGrid-Dynamic-Obstacles-5x5-v0')
#make("MiniGrid-Dynamic-Obstacles-16x16-v0")

class MiniGrid_Dynamic_Obstacles_16x16_v0(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

    def get_actionspace(self):
        print("##", self.env.action_dim)
        return self.env.action_dim

    def get_observationspace(self):
        print("##", self.env.input_dimension)
        return self.env.input_dimension

    def step(self, action, train=True):
        # action = action_wrapper([action])
        next_state, reward, done, _, _ = self.env.step(action)
        reward = np.array(reward, dtype=np.float)
        # reward *= 100
        if reward < 0:
            reward = np.array([0.], dtype=np.float)
        if reward > 0:
            reward *=100
        # if not done:
        #     reward -= 0.5
        next_state = next_state[0]['obs']['image']   #.reshape(60, 80, 3)
        return [{"obs": next_state, "controlled_player_index": 0}], reward, done, _, _

    def reset(self):
        state = self.env.reset()
        state = np.array(state[0]["obs"]['image'])
        return [{"obs": state, "controlled_player_index": 0}]

    def close(self):
        pass

    def set_seed(self, seed):
        self.env.set_seed(seed)



