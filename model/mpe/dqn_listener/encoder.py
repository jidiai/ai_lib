from gym.spaces import Box, Discrete
import numpy as np
import gym

from pettingzoo.mpe import simple_speaker_listener_v3
# from pettingzoo.mpe import simple_v2

env = simple_speaker_listener_v3.parallel_env()
# env = simple_v2.parallel_env()

class Encoder:
    def __init__(self, action_spaces=env.action_space('listener_0'),
                 observation_spaces=env.observation_space('listener_0')):

        self._action_space = action_spaces
        self._observation_space = observation_spaces

    def encode(self, state):
        # obs=np.array([self._policy.state_index(state)],dtype=int)
        # print(self._policy.state_index(state))
        obs = state
        action_mask = np.ones(self._action_space.n, dtype=np.float32)
        return obs, action_mask

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
