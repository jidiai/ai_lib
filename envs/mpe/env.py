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

class GlobalFeatureEncoder:
    def __init__(self, action_spaces_dict, observation_spaces_dict):
        self._action_space_dict = action_spaces_dict
        self._observation_space_dict = observation_spaces_dict
        self._agent_ids = list(self._observation_space_dict.keys())

    def encode(self, global_state):
        encoded_obs = {}
        action_masks = {}
        for aid in self._agent_ids:
            current_obs = global_state[aid]
            other_obs = [global_state[i] for i in self._agent_ids if i != aid]
            other_obs = np.concatenate(other_obs)
            current_encoded_obs = np.concatenate([current_obs, other_obs])
            encoded_obs[aid] = current_encoded_obs
            action_masks[aid] = np.ones(self._action_space_dict[aid].n, dtype=np.float32)
        return encoded_obs, action_masks



class ParameterSharingFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces):
        self._action_space = action_spaces
        self._observation_space = observation_spaces

    def encode(self, state):
        share_obs = [state[agent_id] for agent_id in state]
        action_mask = [np.ones(self._action_space.n, dtype=np.float32)]
        return np.stack(share_obs), np.stack(action_mask)



@registry.registered(registry.ENV, "mpe")
class MPE(BaseAECEnv):
    def __init__(self, id, seed, cfg):
        self.id = id
        self.seed = seed
        self.cfg = cfg
        env_id = cfg["env_id"]
        env_module = importlib.import_module(f"pettingzoo.mpe.{env_id}")
        self._env = env_module.parallel_env()            #max_cycles=25, continuous_actions=False
        self._env.seed(seed)
        self._step_ctr = 0
        self._is_terminated = False

        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

        self.agent_ids = self._env.possible_agents    #["agent_0", "agent_1"]
        self.num_players = dict(zip(self.agent_ids, [1]*len(self.agent_ids)))
            #{"agent_0": 1, "agent_1": 1}
        # self.feature_encoders = {
        #     "agent_0": DefaultFeatureEncoder(self._action_space['agent_0'], self._observation_space['agent_0']),
        #     "agent_1": DefaultFeatureEncoder(self._action_space['agent_0'], self._observation_space['agent_0']),
        # }
        self.global_encoder = self.cfg['global_encoder']
        if not self.global_encoder:
            self.feature_encoders = {
                aid: DefaultFeatureEncoder(self._action_space(aid), self._observation_space(aid))
                for aid in self.agent_ids
            }
        else:
            self.feature_encoders = GlobalFeatureEncoder(self._env.action_spaces,
                                                         self._env.observation_spaces)


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
        # if 'feature_encoders' in  custom_reset_config:
        #     self.feature_encoders = custom_reset_config['feature_encoders']
        self._step_ctr = 0
        self._is_terminated=False
        # self.cumulated_rewards = 0
        self.stats={agent_id: {
            "total_reward": 0.0,
            "total_steps": 0,
            f"{agent_id}'s reward": 0,
        } for agent_id in self.agent_ids
        }


        observations = self._env.reset()  # {agent_id: agent_obs}
        # if not self.global_encoder:
        if not self.global_encoder:
            encoded_observations = {}
            action_masks = {}
            dones = {}
            for agent_id in self.agent_ids:
                _obs, _action_mask = self.feature_encoders[agent_id].encode(observations[agent_id])
                encoded_observations[agent_id] = np.array(_obs, dtype=np.float32)
                action_masks[agent_id] = np.array(_action_mask)
                dones[agent_id] = np.zeros((1))
        else:
            encoded_observations, action_masks = self.feature_encoders.encode(observations)
            dones = dict(zip(self.agent_ids, [np.zeros((1))]*len(self.agent_ids)))
        # else:
        #     encoded_observations =

        rets = {
            agent_id:{
                EpisodeKey.CUR_OBS: encoded_observations[agent_id],
                EpisodeKey.ACTION_MASK: action_masks[agent_id],
                EpisodeKey.DONE: dones[agent_id]
            }
            for agent_id in self.agent_ids
        }
        return rets


    def step(self, actions):
        """
        actions: Dict[agent_id: Dict['action': int]]
        """
        self._step_ctr+=1

        filtered_actions = {
            aid: int(actions[aid]) for aid in self.agent_ids
        }

        # for agent, action in actions.items():
        #     assert self.action_spaces[agent].contains(action), f"Action is not in space: {action} with type={type(action)}"
        observations, rewards, dones, infos = self._env.step(filtered_actions)

        if not self.global_encoder:
            encoded_observations = {}
            action_masks = {}
            for agent_id in self.agent_ids:
                _obs, _action_mask = self.feature_encoders[agent_id].encode(observations[agent_id])
                encoded_observations[agent_id] = np.array(_obs, dtype=np.float32)
                action_masks[agent_id] = np.array(_action_mask)
        else:
            encoded_observations, action_masks = self.feature_encoders.encode(observations)

        self._is_terminated=all(list(dones.values()))
        self.update_episode_stats(rewards)


        return {
            agent_id: {
                EpisodeKey.NEXT_OBS: encoded_observations[agent_id],
                EpisodeKey.NEXT_ACTION_MASK: np.ones(self.action_spaces('agent_0').n, dtype=np.float32),
                EpisodeKey.REWARD: np.array([rewards[agent_id]]),
                EpisodeKey.DONE: np.array([dones[agent_id]])
            }   for agent_id in self.agent_ids
        }

        # return {
        #     EpisodeKey.NEXT_OBS: observations,
        #     EpisodeKey.REWARD: rewards,
        #     EpisodeKey.DONE: dones,
        #     EpisodeKey.INFO: infos,
        # }

    def render(self, *args, **kwargs):
        self._env.render()

    def update_episode_stats(self, reward):

        for agent_id, stat in self.stats.items():
            stat['total_reward'] += sum(reward.values())  #reward[agent_id]
            stat['total_steps'] = self._step_ctr
            stat[f"{agent_id}'s reward"] += reward[agent_id]

    def get_episode_stats(self):
        return self.stats

    def is_terminated(self):
        return self._is_terminated
    def close(self):
        pass


