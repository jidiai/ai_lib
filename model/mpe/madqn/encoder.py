from gym.spaces import Box, Discrete
import numpy as np
import gym

from pettingzoo.mpe import simple_reference_v2
# from pettingzoo.mpe import simple_v2

env = simple_reference_v2.parallel_env()
# env = simple_v2.parallel_env()
def merge_gym_box(box_list):
    length = len(box_list)
    total_shape = box_list[0].shape[0]
    low = box_list[0].low
    high = box_list[0].high
    dtype = box_list[0].dtype

    for i in range(1,length):
        assert box_list[0] == box_list[i], f"box list has unequal elements, {box_list[0] and box_list[i]}"
        low = np.concatenate([low, low])
        high = np.concatenate([high, high])
        total_shape += box_list[i].shape[0]

    return gym.spaces.Box(low=low,high=high, shape=(total_shape,), dtype =dtype)

state_space = merge_gym_box([env.observation_space(aid)
                             for aid in env.possible_agents])



class Encoder:
    def __init__(self, action_spaces=env.action_space('agent_0'),
                 observation_spaces=state_space):

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

# class GlobalEncoder:
#     def __init__(self, action_spaces = env.action_space('agent_0'),
#                  observation_spaces=state_space):
#         self._action_space = action_spaces
#         self._observation_space = observation_spaces
#
#     def encoder(self, state):
#         pass

