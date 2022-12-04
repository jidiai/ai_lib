from registry import registry
from utils.episode import EpisodeKey
from ..base_aec_env import BaseAECEnv

import numpy as np
import gym
from gym.spaces import Box, Discrete


class DefaultGymFeatureEncoder:
    def __init__(self, env):
        self._env = env

    def encode(self, observation, agent_id):
        obs = observation["observation"]
        action_mask = observation["action_mask"]
        return obs, action_mask

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space


class DefaultGymRewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):
        return np.array([raw_rewards])


@registry.registered(registry.ENV, "gym")
class GymEnv(BaseAECEnv):
    def __init__(self, id, seed, cfg):
        self.id = id
        self.seed = seed
        self.cfg = cfg

        self._env = gym.make(cfg["env_id"])
        self._env.seed(seed)
        self._step_ctr = 0
        self._is_termomated = False

        self.agent_ids = ["agent_0"]
        self.num_players = {"agent_0": 1}
        self.feature_encoders = {"agent_0": DefaultGymFeatureEncoder}

        self.rewarder = DefaultGymRewarder()
        # {
        #     'agent_0': DefaultGymRewarder
        # }

        self._last_record = None

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def id_mapping(self, player_id):
        return player_id.replace("player", "agent")

    def _get_curr_agent_data(self, agent_id):
        observation_all, reward, done, _ = self.env_last()
        self._last_observation = observation_all["observation"]
        observation, action_mask = self.feature_encoders[agent_id].encode(
            observation_all, agent_id
        )

        return {
            agent_id: {
                EpisodeKey.CUR_OBS: np.array([observation]),
                EpisodeKey.ACTION_MASK: np.array([action_mask]),
                EpisodeKey.REWARD: np.array([[reward]]),
                EpisodeKey.DONE: np.array([[done]]),
            }
        }

    def env_last(self):
        return self._last_record

    def reset(self, custom_reset_config=None):
        if (
            custom_reset_config is not None
            and "feature_encoders" in custom_reset_config
        ):
            self.feature_encoders = custom_reset_config["feature_encoders"]
        if custom_reset_config is not None and "rewarder" in custom_reset_config:
            self.rewarder = custom_reset_config["rewarders"]["agent_0"]

        self._step_ctr = 0
        self._is_terminated = False
        self._last_agent_id = None
        self._last_action = None
        self.stats = {agent_id: {"reward": 0.0} for agent_id in self.agent_ids}
        self.cumulated_rewards = 0

        init_obs = self._env.reset()
        dones = {k: np.zeros((v), dtype=bool) for k, v in self.num_players.items()}

        if isinstance(self.action_space, Discrete):
            action_mask = np.ones(self.action_space.n, dtype=np.float32)
        elif isinstance(self.action_space, Box):
            action_mask = np.ones(self.action_space.shape)
        else:
            action_mask = None

        rets = {
            agent_id: {
                EpisodeKey.CUR_OBS: init_obs,
                EpisodeKey.ACTION_MASK: action_mask,
                EpisodeKey.DONE: dones[agent_id],
            }
            for agent_id in self.agent_ids
        }
        return rets

    def agent_id_iter(self):
        return self.agent_ids[0]

    @property
    def step_ctr(self):
        return self._step_ctr

    # @property
    def is_terminated(self):
        return self._is_terminated

    def get_curr_agent_data(self, agent_id):
        data = self._get_curr_agent_data(agent_id)
        self._last_agent_id = agent_id

        self._update_episode_stats(data)
        return data

    def step(self, actions):
        self._step_ctr += 1
        # print('steps = ', self._step_ctr)
        assert len(actions) == 1

        action = list(actions.values())[0]["action"]
        if self._is_terminated:
            assert action is None, "{} {}".format(self._is_terminated, actions)
        else:
            action = int(action)
            if isinstance(self.action_space, Box):
                action = np.array([action])

        observation, _reward, done, info = self._env.step(action)

        rewards = self.rewarder.r(_reward, obs=observation)
        # if observation[0] > -0.5:
        #     r = observation[0] + 0.5 - 1
        #     if observation[0] > 0.5:
        #         r = 300
        # else:
        #     r = -1.
        # rewards = np.array([r])

        # rewards = np.array([rewards])
        self._is_terminated = done
        self.update_episode_stats(rewards)

        # self._last_action=action
        dones = {
            k: np.full((v), fill_value=done, dtype=bool)
            for k, v in self.num_players.items()
        }

        if isinstance(self.action_space, Discrete):
            action_mask = np.ones(self.action_space.n, dtype=np.float32)
        elif isinstance(self.action_space, Box):
            action_mask = np.ones(self.action_space.shape)
        else:
            action_mask = None

        rets = {
            agent_id: {
                EpisodeKey.NEXT_OBS: observation,
                EpisodeKey.NEXT_ACTION_MASK: action_mask,
                EpisodeKey.REWARD: rewards,
                EpisodeKey.DONE: dones[agent_id],
            }
            for agent_id in self.agent_ids
        }
        return rets

    def update_episode_stats(self, reward):
        reward = float(reward)
        self.cumulated_rewards += reward

        self.stats["agent_0"] = {
            "reward": self.cumulated_rewards,
            "total_steps": self._step_ctr,
        }

    def get_episode_stats(self):
        return {"agent_0": self.stats["agent_0"]}

    def render(self, mode="cmd"):
        self._env.render()
