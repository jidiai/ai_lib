from ...default.ppo_actor import Actor
from ...default.ppo_critic import Critic
from ...default.encoder import Encoder as encoder_cls

import numpy as np

from pettingzoo.mpe import simple_speaker_listener_v3
env = simple_speaker_listener_v3.parallel_env()

class Encoder(encoder_cls):
    def __init__(self):
        super().__init__(action_spaces=env.action_space('listener_0'),
                         observation_spaces=env.observation_space('listener_0'),
                         state_space=env.observation_space('listener_0'))
class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):

        return np.array([raw_rewards])
