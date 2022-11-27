from .actor import Actor
from .critic import Critic
from .encoder import Encoder

import numpy as np


class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):
        return np.array([raw_rewards])
