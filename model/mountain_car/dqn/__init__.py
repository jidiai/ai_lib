from .actor import Actor
from .critic import Critic
from .encoder import Encoder

import numpy as np


class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):
        obs = kwargs["obs"]
        if obs[0] > -0.5:
            r = obs[0] + 0.5 - 1
            if obs[0] > 0.5:
                r = 100
        else:
            r = -1.0
        rewards = np.array([r])

        return np.array([rewards])
