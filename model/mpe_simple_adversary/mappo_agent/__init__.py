from ...default.ppo_actor import Actor
from ...default.ppo_critic import Critic
from ...default.encoder import Encoder as encoder_cls

import numpy as np
import gym

from pettingzoo.mpe import simple_adversary_v2
env = simple_adversary_v2.parallel_env()


def merge_gym_box(box_list):
    length = len(box_list)
    total_shape = box_list[0].shape[0]
    low = box_list[0].low
    high = box_list[0].high
    dtype = box_list[0].dtype

    for i in range(1,length):
        low = np.concatenate([low, box_list[i].low])
        high = np.concatenate([high, box_list[i].high])
        total_shape += box_list[i].shape[0]

    return gym.spaces.Box(low=low,high=high, shape=(total_shape,), dtype =dtype)

_state_space = merge_gym_box([env.observation_spaces[aid]
                             for aid in ['agent_0', 'agent_1']])


class Encoder(encoder_cls):
    def __init__(self):
        super().__init__(action_spaces=env.action_spaces['agent_0'],
                         observation_spaces=env.observation_spaces['agent_0'],
                         state_space=_state_space)


class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):

        return np.array([raw_rewards])
