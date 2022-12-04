import importlib
import gym
from registry import registry
from utils.episode import EpisodeKey

from ..base_aec_env import BaseAECEnv
from utils.episode import EpisodeKey
import numpy as np


class DefaultFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces):

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

class MPE(BaseAECEnv):
    def __init__(self, id, seed, cfg):
        self.id = id
        self.seed = seed
        self.cfg = cfg
        env_id = cfg["env_id"]
        env_module = importlib.import_module(f"pettingzoo.mpe.{env_id}")
        self._env = env_module.parallel_env()            #max_cycles=25, continuous_actions=False
        self._step_ctr = 0
        self._is_terminated = False

        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

        self.agent_ids = self._env.possible_agents    #["agent_0", "agent_1"]
        self.num_players = {"agent_0": 1, "agent_1": 1}
        self.feature_encoders = {
            "agent_0": DefaultFeatureEncoder(self._action_space['agent_0'], self._observation_space['agent_0']),
            "agent_1": DefaultFeatureEncoder(self._action_space['agent_0'], self._observation_space['agent_0']),
        }
        self.max_step = 25


    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def action_spaces(self):
        return self._action_space

    @property
    def observation_spaces(self):
        return self._observation_space

    def reset(self, custom_reset_config=None):
        self.feature_encoders = custom_reset_config['feature_encoders']
        self.step_ctr = 0
        observations = self._env.reset()  # {agent_id: agent_obs}
        encoded_observations = {}
        action_masks = {}
        dones = {}
        for agent_id in self.agent_ids:
            _obs, _action_mask = self.feature_encoders[agent_id].encode(observations[agent_id])
            encoded_observations[agent_id] = np.array(_obs, dtype=np.float32)
            action_masks[agent_id] = np.array(_action_mask)
            dones[agent_id] = np.zeros((1))

        rets = {
            agent_id:{
                EpisodeKey.CUR_OBS: encoded_observations[agent_id],
                EpisodeKey.ACTION_MASK: action_masks[agent_id],
                EpisodeKey.DONE: dones[agent_id]
            }
            for agent_id in self.agent_ids
        }
        return rets


    def time_step(self, actions):
        # for agent, action in actions.items():
        #     assert self.action_spaces[agent].contains(action), f"Action is not in space: {action} with type={type(action)}"
        observations, rewards, dones, infos = self._env.step(actions)
        return {
            EpisodeKey.CUR_OBS: observations,
            EpisodeKey.REWARD: rewards,
            EpisodeKey.DONE: dones,
            EpisodeKey.INFO: infos,
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def close(self):
        pass


