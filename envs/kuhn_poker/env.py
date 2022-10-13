from utils.episode import EpisodeKey

import numpy as np
from ..base_aec_env import BaseAECEnv
from .raw_env import KuhnPokerRawEnv
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
from gym.spaces import Box,Discrete
from registry import registry

class DefaultFeatureEncoder:
    def __init__(self):
        game = pyspiel.load_game('kuhn_poker')
        self._policy=policy_lib.TabularPolicy(game)

    def encode(self,state):
        #obs=np.array([self._policy.state_index(state)],dtype=int)
        # print(self._policy.state_index(state))
        obs=np.array(state.observation_tensor())
        legal_action_idices=state.legal_actions()
        action_mask=np.zeros(2,dtype=np.float32)
        action_mask[legal_action_idices]=1
        return obs,action_mask

    @property
    def observation_space(self):
        return Box(0.0,1.0,shape=(11,))

    @property
    def action_space(self):
        return Discrete(2)

@registry.registered(registry.ENV,"kuhn_poker")
class KuhnPokerEnv(BaseAECEnv):
    '''
    open_spiel:
        observation_space Box(11)
        action_space Discrete(2) 0-pass,1-bet
    See what observation means:
        https://github.com/deepmind/open_spiel/blob/master/open_spiel/integration_tests/playthroughs/kuhn_poker_2p.txt
    '''
    def __init__(self,id,seed,cfg):
        self.id=id
        self.seed=seed
        self.cfg=cfg

        self._env=KuhnPokerRawEnv(seed)
        self._step_ctr=0
        self._is_terminated=False

        self.agent_ids=["agent_0","agent_1"]
        self.num_players={
            "agent_0": 1,
            "agent_1": 1
        }
        self.feature_encoders={
            "agent_0": DefaultFeatureEncoder(),
            "agent_1": DefaultFeatureEncoder()
        }

    def id_mapping(self,player_id):
        return player_id.replace("player","agent")

    def _get_curr_agent_data(self,agent_id):
        # NOTE state(here,observation) has full information, we should only pick observable ones.
        observation, reward, done, _ = self._env.last()

        self._last_observation=observation

        observation,action_mask=self.feature_encoders[agent_id].encode(observation)

        return {
            agent_id: {
                EpisodeKey.CUR_OBS: np.array([observation]),
                EpisodeKey.ACTION_MASK: np.array([action_mask]),
                EpisodeKey.REWARD: np.array([[reward]]),
                EpisodeKey.DONE: np.array([[done]])
            }
        }

    def reset(self,custom_reset_config=None):
        if custom_reset_config is not None and "feature_encoders" in custom_reset_config:
            self.feature_encoders=custom_reset_config["feature_encoders"]

        self._step_ctr=0
        self._is_terminated=False
        self._last_agent_id=None
        self._last_action=None
        self.stats={
            agent_id: {"score": 0.0, "reward": 0.0}
            for agent_id in self.agent_ids
        }

        self._env.reset()

    def agent_id_iter(self):
        return self._env.agent_iter()

    @property
    def step_ctr(self):
        return self._step_ctr

    @property
    def is_terminated(self):
        return self._is_terminated

    def get_curr_agent_data(self,agent_id):
        data=self._get_curr_agent_data(agent_id)

        self._last_agent_id=agent_id

        self._update_episode_stats(data)
        return data

    def step(self,actions):
        self._step_ctr+=1

        assert len(actions)==1
        action=list(actions.values())[0]
        if self._is_terminated:
            assert action is None,"{} {}".format(self._is_terminated,actions)
        else:
            action=int(action)

        self._env.step(action)

        self._is_terminated=np.all(self._env.dones)

        self._last_action=action

    def get_episode_stats(self):
        return self.stats

    def _update_episode_stats(self,data):
        for agent_id,d in data.items():
            reward=float(d[EpisodeKey.REWARD])
            if reward>0:
                self.stats[agent_id]={
                    "win": 1,
                    "lose": 0,
                    "score": 1,
                    "reward": reward
                }
            elif reward<0:
                self.stats[agent_id]={
                    "win": 0,
                    "lose": 1,
                    "score": 0,
                    "reward": reward
                }
            else:
                self.stats[agent_id]={
                    "win": 0,
                    "lose": 0,
                    "score": 0.5,
                    "reward": reward
                }



    def render(self,mode="cmd"):
        self._env.render()
