import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
from gym.spaces import Box, Discrete
import numpy as np


class Encoder:
    def __init__(self):
        self.observation_space = Box(low=0.0, high=10.0, shape=(11,))
        self.action_space = Discrete(2)

    def encode(self, state):
        # use open_spiel default state encoding
        if state.current_player() in [0, 1]:
            obs = np.array(state.information_state_tensor(state.current_player()))
        else:
            # the game already ends
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        legal_action_idices = state.legal_actions()
        action_mask = np.zeros(self.action_space.n, dtype=np.float32)
        action_mask[legal_action_idices] = 1
        return obs, action_mask
