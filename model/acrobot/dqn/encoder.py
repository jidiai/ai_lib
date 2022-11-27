
from gym.spaces import Box,Discrete
import numpy as np
import gym

cartpole_env = gym.make('Acrobot-v1')

class Encoder:
    def __init__(self):
        self.action_space=Discrete(2)
        self.observation_space = cartpole_env.observation_space
    def encode(self,state):

        return state, np.ones(self.action_space.n, dtype=np.float32)
