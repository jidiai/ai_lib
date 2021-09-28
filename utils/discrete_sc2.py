# -*- coding:utf-8  -*-
# Time  : 2021/9/24 上午10:57
# Author: Yahui Cui


from pysc2.lib import actions
import numpy as np


class Discrete_SC2(object):

    def __init__(self, available_actions, action_spec):
        self.available_actions = available_actions
        self.action_spec = action_spec

    def sample(self):
        function_id = np.random.choice(self.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        action = actions.FunctionCall(function_id, args)
        return action


