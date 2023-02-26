from ...default.dqn_actor import Actor
from ...default.dqn_critic import Critic
from ...default.encoder import Encoder as encoder_cls

import numpy as np

from pettingzoo.mpe import simple_reference_v2
env = simple_reference_v2.parallel_env()

class Encoder(encoder_cls):
    def __init__(self):
        super().__init__(action_spaces=env.action_spaces['agent_0'],
                         observation_spaces=env.observation_spaces['agent_0'],
                         state_space=env.observation_spaces['agent_0'])
class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):

        return np.array([raw_rewards])
