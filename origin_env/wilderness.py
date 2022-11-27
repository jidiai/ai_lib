# -*- coding:utf-8  -*-
# Time  : 2022/6/28 下午4:35
# Author: Yahui Cui
import copy
import random

import gym
import numpy as np
from inspirai_fps.utils import get_distance, get_position
from inspirai_fps.gamecore import Game, ActionVariable

from env.simulators.game import Game as JidiGame
from utils.box import Box
from utils.discrete import Discrete


class Wilderness(JidiGame):
    def __init__(self, conf, seed=0):
        super(Wilderness, self).__init__(
            conf["n_player"],
            conf["is_obs_continuous"],
            conf["is_act_continuous"],
            conf["game_name"],
            conf["agent_nums"],
            conf["obs_type"],
        )
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed
        self.set_seed()
        self.env_core = None
        self.game_config = {}
        if self.game_name == "wilderness-navigation":
            self.game_config = {
                "timeout": 60 * 2,
                "time_scale": 10,
                "map_id": random.randint(1, 10),
                "random_seed": self.seed,
                "target_location": [0, 0, 0],
                "start_location": [0, 0, 0],
                "start_range": 2,
                "start_hight": 5,
                "engine_dir": "../wilderness-scavenger/fps_linux",
                "map_dir": "../wilderness-scavenger/map_data",
                "num_workers": 0,
                "eval_interval": None,
                "record": False,
                "replay_suffix": "",
                "checkpoint_dir": "checkpoints_track1",
                "detailed_log": False,
                "stop_iters": 9999,
                "stop_timesteps": 100000000,
                "stop-reward": 95,
            }
            self.env_core = NavigationEnvSimple(self.game_config)

        self.joint_action_space = self.env_core.action_space
        self.done = False
        self.init_info = None
        self.step_cnt = 0
        self.current_state = self.env_core.reset()
        self.all_observes = self.get_all_observes()
        self.won = ""
        self.n_return = [0]

    def reset(self):
        if self.game_name == "wilderness-navigation":
            self.env_core = NavigationEnvSimple(self.game_config)
        obs = self.env_core.reset()
        self.current_state = obs
        self.done = False
        all_observes = self.get_all_observes()
        return all_observes

    def reset_episode(self):
        """
        use this when training with many episodes together with function
        is_terminal_episode()
        """
        obs = self.env_core.reset()
        self.current_state = obs
        self.done = False
        all_observes = self.get_all_observes()
        return all_observes

    def step(self, joint_action):
        self.step_cnt += 1
        decoded_action = self.decode(joint_action)
        obs, reward, self.done, info = self.env_core.step(decoded_action)
        self.set_n_return(reward)
        self.current_state = obs
        all_observes = self.get_all_observes()
        # print("reward is {}".format(reward))
        return all_observes, reward, self.done, info, ""

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {
                "obs": copy.deepcopy(self.current_state),
                "controlled_player_index": i,
            }
            all_observes.append(each)

        return all_observes

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def decode(self, joint_action):
        decoded_action = []
        for idx, action in enumerate(joint_action[0]):
            if isinstance(self.joint_action_space[0][idx], Discrete):
                decoded_action.append(action.index(1))
            elif isinstance(self.joint_action_space[0][idx], Box):
                decoded_action.append(action[0])
        return decoded_action

    def is_terminal(self):
        if self.done:
            self.env_core.game.close()
        return self.done

    def is_terminal_episode(self):
        return self.done

    def check_win(self):
        return ""

    def set_n_return(self, reward):
        for i in range(self.n_player):
            self.n_return[i] += reward

    def set_seed(self, seed=None):
        if not seed:
            seed = self.seed
        else:
            self.seed = seed
        random.seed(seed)


BASE_WORKER_PORT = 50051


class BaseEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self.record = config.get("record", False)
        self.replay_suffix = config.get("replay_suffix", "")
        self.print_log = config.get("detailed_log", False)

        self.seed(config["random_seed"])
        self.server_port = BASE_WORKER_PORT

        self.game = Game(
            map_dir=config["map_dir"],
            engine_dir=config["engine_dir"],
            server_port=self.server_port,
        )
        self.game.set_map_id(config["map_id"])
        self.game.set_episode_timeout(config["timeout"])
        self.game.set_random_seed(config["random_seed"])
        self.start_location = config.get("start_location", [0, 0, 0])

    def reset(self):
        print("Reset for a new game ...")
        self._reset_game_config()
        if self.record:
            self.game.turn_on_record()
        else:
            self.game.turn_off_record()
        self.game.set_game_replay_suffix(self.replay_suffix)
        self.game.new_episode()
        self.state = self.game.get_state()
        self.running_steps = 0
        return self._get_obs()

    def close(self):
        self.game.close()
        return super().close()

    def render(self, mode="replay"):
        return None

    def _reset_game_config(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()


class NavigationBaseEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

        self.start_range = config["start_range"]
        self.start_hight = config["start_hight"]
        self.trigger_range = self.game.target_trigger_distance
        self.target_location = config["target_location"]

        self.game.set_game_mode(Game.MODE_NAVIGATION)
        self.game.set_target_location(self.target_location)

    def _reset_game_config(self):
        self.start_location = self._sample_start_location()
        self.game.set_start_location(self.start_location)

    def step(self, action):
        action_cmd = self._action_process(action)
        self.game.make_action({0: action_cmd})
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()
        reward = 0
        self.running_steps += 1

        if done:
            cur_pos = get_position(self.state)
            tar_pos = self.target_location

            if get_distance(cur_pos, tar_pos) <= self.trigger_range:
                reward += 100

            if self.print_log:
                Start = np.round(np.asarray(self.start_location), 2).tolist()
                Target = np.round(np.asarray(self.target_location), 2).tolist()
                End = np.round(np.asarray(get_position(self.state)), 2).tolist()
                Step = self.running_steps
                Reward = reward
                print(f"{Start=}\t{Target=}\t{End=}\t{Step=}\t{Reward=}")

        return self._get_obs(), reward, done, {}

    def _sample_start_location(self):
        raise NotImplementedError()

    def _action_process(self, action):
        raise NotImplementedError()


class NavigationEnvSimple(NavigationBaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.action_pools = {
            ActionVariable.WALK_DIR: [0, 90, 180, 270],
            ActionVariable.WALK_SPEED: [3, 6],
            ActionVariable.TURN_LR_DELTA: [-1, 0, 1],
            ActionVariable.LOOK_UD_DELTA: [-1, 0, 1],
            ActionVariable.JUMP: [True, False],
        }
        # self.action_space = MultiDiscrete([len(pool) for pool in self.action_pools.values()])
        self.action_space = [
            [Discrete(len(pool)) for pool in self.action_pools.values()]
        ]
        self.action_space[0][0] = Box(high=360, low=0, shape=(1,), dtype=np.float32)
        self.action_space[0][1] = Box(high=10, low=0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.game.set_available_actions(
            [action_name for action_name in self.action_pools.keys()]
        )
        self.game.init()

    def _get_obs(self):
        cur_pos = np.asarray(get_position(self.state))
        tar_pos = np.asarray(self.target_location)
        dir_vec = tar_pos - cur_pos
        return dir_vec / np.linalg.norm(dir_vec)

    def _action_process(self, action):
        action_values = list(self.action_pools.values())
        final_action = []
        for idx, single_action in enumerate(action):
            if isinstance(self.action_space[0][idx], Discrete):
                final_action.append(action_values[idx][single_action])
            elif isinstance(self.action_space[0][idx], Box):
                final_action.append(single_action)
        # return [action_values[i][action[i]] for i in range(len(action))]
        return final_action

    def _sample_start_location(self):
        angle = np.random.uniform(0, 360)
        distance_to_trigger = abs(np.random.normal(scale=self.start_range))
        vec_len = self.trigger_range + distance_to_trigger
        dx = np.sin(angle) * vec_len
        dz = np.cos(angle) * vec_len
        x = self.target_location[0] + dx
        z = self.target_location[2] + dz
        return [x, self.start_hight, z]
