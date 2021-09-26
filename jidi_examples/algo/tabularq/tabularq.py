import numpy as np

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer
from common.utils import plot_action_values


def get_trajectory_property():
    return ["action"]


class TABULARQ(object):

    def __init__(self, args):

        self.args = args
        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.buffer_size = args.buffer_capacity
        self.gamma = args.gamma
        self.lr = args.lr
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)

        # define initial Q table
        self._q = np.zeros((self.state_dim, self.action_dim))

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

    @property
    def q_values(self):
        return self._q

    def behaviour_policy(self, q):
        self.eps = max(self.eps_end, self.eps - self.eps_delay)
        return self.epsilon_greedy(q, epsilon=self.eps)

    def target_policy(self, q):
        return np.eye(len(q))[np.argmax(q)]

    def epsilon_greedy(self, q_values, epsilon):
        if epsilon < np.random.random():
            return np.argmax(q_values)
        else:
            return np.random.randint(np.array(q_values).shape[-1])

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train):
        action = self.behaviour_policy(self.q_values[observation, :])
        return {"action": action}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):
        data = self.memory.get_step_data()

        next_state = data['states_next']
        state = data['states']
        reward = data['rewards']
        action = data['action']
        done = data['dones']

        target_index = self.target_policy(self._q[next_state, :])
        target = reward + self.gamma * (self._q[next_state, :] @ target_index) * (1 - done)
        self._q[state, action] += self.lr * (target - self._q[state, action])

    def save(self, save_path, episode):
        if self.args.scenario == "gridworld":
            W = -100  # wall
            G = 100  # goal
            GRID_LAYOUT = np.array([
                [W, W, W, W, W, W, W, W, W, W, W, W],
                [W, W, 0, W, W, W, W, W, W, 0, W, W],
                [W, 0, 0, 0, 0, 0, 0, 0, 0, G, 0, W],
                [W, 0, 0, 0, W, W, W, W, 0, 0, 0, W],
                [W, 0, 0, 0, W, W, W, W, 0, 0, 0, W],
                [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W],
                [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W],
                [W, W, 0, 0, 0, 0, 0, 0, 0, 0, W, W],
                [W, W, W, W, W, W, W, W, W, W, W, W]
            ])
            plot_action_values(self.args.algo, GRID_LAYOUT, self._q.reshape((9, 12) + (4,)), vmin=-20, vmax=100)

