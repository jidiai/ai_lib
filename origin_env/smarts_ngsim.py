# -*- coding:utf-8  -*-
# Time  : 2021/10/13 下午4:47
# Author: Yahui Cui


import os
import sys
from pathlib import Path

CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
CURRENT_FOLDER = str(Path(__file__).resolve().parent.parent)
smarts_path = os.path.join(CURRENT_PATH, "SMARTS")
sys.path.append(smarts_path)
sys.setrecursionlimit(1000000)
import pickle
import random
import numpy as np
from dataclasses import replace


from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType


from env.simulators.game import Game
from utils.box import Box


class SmartsNGSIM(Game):
    def __init__(self, conf):
        super(SmartsNGSIM, self).__init__(
            conf["n_player"],
            conf["is_obs_continuous"],
            conf["is_act_continuous"],
            conf["game_name"],
            conf["agent_nums"],
            conf["obs_type"],
        )
        self.max_step = int(conf["max_step"])
        self.partial = conf["partial"]
        scenario_name = conf["scenario_name"]
        self.scenario_path = os.path.join(
            CURRENT_PATH, "SMARTS", "scenarios", scenario_name
        )
        test_file_path = os.path.join(
            CURRENT_FOLDER, "env", "ngsim_jidi", "test_data.pkl"
        )

        with open(test_file_path, "rb") as f:
            self.test_ids = pickle.load(f)

        if self.partial:
            sample_num = 10
            sample_list = [key for key in self.test_ids.keys()]
            sample_list = random.sample(sample_list, sample_num)
            temp_data = {}
            for key in sample_list:
                temp_data[key] = self.test_ids[key]
            self.test_ids = temp_data

        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space
        self.info = {"finish": False}
        self.reset()

    def reset(self):
        self.env_core = SMARTSImitation([self.scenario_path], self.test_ids)
        self.step_cnt = 0
        self.step_cnt_per_vehicle = 0
        self.done = False
        self.init_info = None
        obs = self.env_core.reset()
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.won = {}
        self.n_return = [0] * self.n_player
        return self.all_observes

    def step(self, joint_action):
        info_before = ""
        self.is_valid_action(joint_action)
        action = self.decode(joint_action)
        obs, reward, self.done, self.info = self.env_core.step(action)
        self.current_state = obs
        self.all_observes = self.get_all_observes()
        self.step_cnt += 1
        self.step_cnt_per_vehicle += 1
        if self.step_cnt_per_vehicle >= self.max_step:
            self.done = True
        reward = self.get_reward(reward)
        done = self.is_terminal()
        if done:
            self.set_final_n_return()
        info_after = self.info
        return self.all_observes, reward, done, info_before, info_after

    def get_all_observes(self):
        all_observes = []
        each = {"obs": self.current_state, "controlled_player_index": 0}
        all_observes.append(each)
        return all_observes

    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:
            raise Exception(
                "Input joint action dimension should be {}, not {}".format(
                    self.n_player, len(joint_action)
                )
            )

        if (not isinstance(joint_action[0], list)) and (
            not isinstance(joint_action[0], np.ndarray)
        ):
            raise Exception(
                "Submitted action should be list or np.ndarray, not {}.".format(
                    type(joint_action[0])
                )
            )

        action_shape = (
            np.array(joint_action).shape
            if not isinstance(joint_action, np.ndarray)
            else joint_action.shape
        )
        if len(action_shape) != 3:
            raise Exception(
                "joint action shape should be in length 3: (1, 1, {}), not the length of {}.".format(
                    self.joint_action_space[0][0].shape[0], len(action_shape)
                )
            )
        if (
            action_shape[0] != 1
            or action_shape[1] != 1
            or action_shape[2] != self.joint_action_space[0][0].shape[0]
        ):
            raise Exception(
                "joint action shape should be (1, 1, {}), not {}.".format(
                    self.joint_action_space[0][0].shape[0], action_shape
                )
            )

    def get_reward(self, reward):
        r = [0] * self.n_player
        # print("reward is ", reward)
        for i in range(self.n_player):
            r[i] = reward
            if self.done:
                self.n_return[i] += r[i]
        return r

    def set_final_n_return(self):
        for i in range(self.n_player):
            self.n_return[i] /= len(self.test_ids)

    def decode(self, joint_action):
        return joint_action[0][0]

    def is_terminal(self):
        if self.info["finish"]:
            return True
        elif self.done:
            obs = self.env_core.reset()
            self.current_state = obs
            self.all_observes = self.get_all_observes()
            self.done = False
            self.step_cnt_per_vehicle = 0
            return False
        else:
            return False

    def set_action_space(self):
        return [[Box(-2.5, 2.5, shape=(2,))] for i in range(self.n_player)]

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def check_win(self):
        return ""


class SMARTSImitation:
    def __init__(self, scenarios, vehicle_data):
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._init_scenario()
        self.vehicle_demo_data = vehicle_data
        self.vehicle_ids = list(vehicle_data.keys())
        self.vehicle_num = len(self.vehicle_ids)
        self.obs_stacked_size = 1
        self.agent_spec = get_agent_spec(self.obs_stacked_size)
        self._vehicle_step = 0

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=None,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def _compute_reward(self, observations, dones):
        ego_pos = observations[self.vehicle_id].ego_vehicle_state.position[:2]
        self.single_traj.append(ego_pos)
        demo_total_step = len(self.vehicle_demo_data[self.vehicle_id])
        if self._vehicle_step < demo_total_step:
            # if ego vehicle terminates before the expert demo, we still need to take
            # into account the additional penalty of expert's remaining timesteps.
            if dones[self.vehicle_id]:
                distance = 0
                for ts in range(self._vehicle_step, demo_total_step):
                    distance += np.linalg.norm(
                        ego_pos - self.vehicle_demo_data[self.vehicle_id][ts]
                    )
                return -distance
            demo_pos = self.vehicle_demo_data[self.vehicle_id][self._vehicle_step]
            demo_traj = self.vehicle_demo_data[self.vehicle_id][: self._vehicle_step]
        else:
            demo_pos = self.vehicle_demo_data[self.vehicle_id][-1]
            demo_traj = self.vehicle_demo_data[self.vehicle_id][:]
        distance = np.linalg.norm(ego_pos - demo_pos)
        frechet_distance = compute_frechet_distance(self.single_traj, demo_traj)
        # return -distance
        return -frechet_distance

    def step(self, action):

        observations, rewards, dones, _ = self.smarts.step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)}
        )
        self._vehicle_step += 1

        rewards[self.vehicle_id] = self._compute_reward(observations, dones)

        if dones[self.vehicle_id] and self.vehicle_itr == self.vehicle_num:
            info = dict(finish=True)
        else:
            info = dict(finish=False)

        return (
            observations[self.vehicle_id],
            rewards[self.vehicle_id],
            dones[self.vehicle_id],
            info,
        )

    def reset(self):
        if self.vehicle_itr >= len(self.vehicle_ids):
            self.vehicle_itr = 0

        self._vehicle_step = 0
        self.single_traj = []
        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts.switch_ego_agents({self.vehicle_id: self.agent_spec.interface})

        observations = self.smarts.reset(self.scenario)
        self.vehicle_itr += 1
        return observations[self.vehicle_id]

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_itr = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()


def compute_frechet_distance(sample_traj, expert_traj):

    frechet_solver = DiscreteFrechet(dist_func=lambda p, q: np.linalg.norm(p - q))
    frechet_distance = frechet_solver.distance(
        np.stack([traj[:2] for traj in expert_traj], axis=0),
        np.concatenate([traj for traj in sample_traj], axis=0),
    )
    return frechet_distance


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])

    return action_adapter


def get_agent_spec(obs_stack_size):

    agent_spec = AgentSpec(
        interface=AgentInterface(
            max_episode_steps=None,
            waypoints=True,
            neighborhood_vehicles=True,
            ogm=False,
            rgb=False,
            lidar=False,
            action=ActionSpaceType.Imitation,
        ),
        action_adapter=get_action_adapter(),
    )

    return agent_spec


class DiscreteFrechet(object):
    """
    Calculates the discrete Fréchet distance between two poly-lines using the
    original recursive algorithm
    """

    def __init__(self, dist_func):
        """
        Initializes the instance with a pairwise distance function.
        :param dist_func: The distance function. It must accept two NumPy
        arrays containing the point coordinates (x, y), (lat, long)
        """
        self.dist_func = dist_func
        self.ca = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines p and q
        This function implements the algorithm described by Eiter & Mannila
        :param p: Poly-line p
        :param q: Poly-line q
        :return: Distance value
        """

        def calculate(i: int, j: int) -> float:
            """
            Calculates the distance between p[i] and q[i]
            :param i: Index into poly-line p
            :param j: Index into poly-line q
            :return: Distance value
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist_func(p[i], q[j])
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i - 1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j - 1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(
                    min(
                        calculate(i - 1, j),
                        calculate(i - 1, j - 1),
                        calculate(i, j - 1),
                    ),
                    d,
                )
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        return calculate(n_p - 1, n_q - 1)
