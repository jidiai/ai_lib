from gym.spaces import Box, Discrete
import numpy as np
import gym

pendulum_env = gym.make("Pendulum-v0")


class Encoder:
    def __init__(self):
        self.action_space = Box(low=-2.0, high=2.0, shape=[1])
        self.observation_space = pendulum_env.observation_space

    def encode(self, state):

        return state, np.ones(self.action_space.n, dtype=np.float32)
