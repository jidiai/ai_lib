from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
print("+",base_dir)
from env.chooseenv import make
env = make('battle_3v3')


class battle_3v3(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

        self.agent2id = self.env.player_id_map
        self.id2agent = self.env.player_id_reverses_map
        self.n_red, self.n_blue = self.env.n_player / 2, self.env.n_player / 2

    def get_actionspace(self):
        """
        return [[Discrete], [Discrete], ...]
        """
        return self.env.joint_action_space

    def get_observationspace(self):
        """
        return [[Box], [Box], ...]
        """
        return [[i] for i in self.env.input_dimension.values()]

    def step(self, joint_action):
        """
        joint_action:   one hot action
        the env_core env will delet reward and done value for agents who died, this wrapper fill in the gap and return
        sorted reward amd done list, the observation follows the convention setting.
        """

        next_state, reward, done, _, _ = self.env.step(joint_action)

        if len(reward) < self.env.n_player:
            for key in list(self.env.player_id_map.keys()):
                if key not in reward.keys():
                    reward[key] = 0
                    done[key] = True
        #sorted
        sorted_reward = dict(sorted({self.env.player_id_map[i]: reward[i] for i in reward.keys()}.items(),
                                    key = lambda x:x[0]))
        sorted_done = dict(sorted({self.env.player_id_map[j]: done[j] for j in done.keys()}.items(),
                                  key = lambda x:x[0]))

        return next_state, list(sorted_reward.values()), list(sorted_done.values()), _, _

    def reset(self):
        state = self.env.reset()
        return state

    def set_seed(self, seed):
        self.env.set_seed(seed)

    def make_render(self):
        self.env.env_core.render()

    def current_agent(self):
        """
        this fn will still remain one last agent on one side when the game terminate, but usable for normal iteration
        """
        agent_list = self.env.env_core.agents  # list of alived agents
        return [self.agent2id[i] for i in agent_list]

    def count_killed(self):
        red_handle, blue_handle = self.env.env_core.env.get_handles()
        red_alive, blue_alive = 0, 0
        # n_red, n_blue = self.env.n_player, self.env.n_player

        a = self.env.env_core.env.get_alive(red_handle)
        red_alive = np.count_nonzero(a + np.zeros(len(a)))

        b = self.env.env_core.env.get_alive(blue_handle)
        blue_alive = np.count_nonzero(b + np.zeros(len(b)))

        return self.n_red - red_alive, self.n_blue - blue_alive

    def minimap(self):
        plt.imshow(self.env.env_core.state().sum(-1).transpose())
        plt.xlabel('x axis')
        plt.ylabel('y axis')