'''
See https://github.com/JBLanier/pipeline-psro/blob/master/multiplayer-rl/mprl/rl/envs/opnspl/poker_multiagent_env.py
'''

from utils.episode import EpisodeKey
from ..base_env import BaseEnv

import numpy as np
from open_spiel.python.rl_environment import TimeStep,Environment
from gym.spaces import Discrete,Box,Dict

class KuhnPokerEnv(BaseEnv):
    '''
    TODO make switch outside.
    '''
    def __init__(self,id,seed,cfg):
        self.id=id
        self.seed=seed
        self.agent_ids=None
        self.step_ctr=0
        self._env=Environment(
            game="kuhn_poker",
            discount=1.0,
            players=2
        )
        
        # TODO(jh): manifest
        self.action_space=Discrete(2)
        self.observation_space=Dict({
            "partial_observation": Box(low=0.0,high=1.0,shape=(11,)),
            "valid_actions_mask": Box(low=0.0,high=1.0,shape=(2,))
        })
        
        
        self.agent_ids=["agent_0","agent_1"]
        self.feature_encoders={
            "agent_0": None,
            "agent_1": None
        }
        
    def reset(self,custom_reset_config):
        # self.feature_encoders=custom_reset_config["feature_encoders"]
        
        self.step_ctr=0
        self.stats={
            agent_id: {"win": 0.0, "lose": 0.0, "score": 0.0, "reward": 0.0}
            for agent_id in self.agent_ids
        }
        
        self.step_data:TimeStep=self._env.reset()
        
        player_id=self.get_current_player_id()
        another_player_id=self.get_another_player_id()
        
        encoded_observations,action_masks=self.encode()
        dones={
            agent_id: np.zeros((1,1),dtype=bool) for agent_id in self.agent_ids
        }
        
        
        return {
            player_id: {
                EpisodeKey.NEXT_OBS: encoded_observations[player_id],
                EpisodeKey.ACTION_MASK: action_masks[player_id],
                EpisodeKey.DONE: dones[player_id]
            },
            another_player_id: {
                EpisodeKey.DONE: dones[another_player_id]
            }
        }
        
    def get_current_agent_ids(self):
        return [self.get_current_player_id()]
        
    def get_current_player_id(self):
        player_idx=self.step_data.observations["current_player"]
        player_id="agent_{}".format(player_idx)
        return player_id
    
    def get_another_player_id(self):
        player_idx=1-self.step_data.observations["current_player"]
        player_id="agent_{}".format(player_idx)
        return player_id
    
    def step(self,actions):
        self.step_ctr+=1
        
        assert len(actions)==1
        actions=[actions[self.get_current_player_id()]]
        self.step_data=self._env.step(actions)
        
        player_id=self.get_current_player_id()
        another_player_id=self.get_another_player_id()

        done=self.step_data.last()
        dones={
            agent_id: np.array([[done]],dtype=bool) for agent_id in self.agent_ids
        }
        
        if not done:
            encoded_observations,action_masks=self.encode()
            assert np.all(np.array(self.step_data.rewards)==0)
            rewards={
                agent_id: np.array([[0]],dtype=np.float32) for agent_id in self.agent_ids
            }
        else:
            assert np.sum(self.step_data.rewards)==0
            rewards={
                "agent_0": np.array([[self.step_data.rewards[0]]],dtype=np.float32),
                "agent_1": np.array([[self.step_data.rewards[1]]],dtype=np.float32),
            }

        self.update_episode_stats(rewards)
        
        if not done:
            ret={ 
                player_id:{
                    EpisodeKey.NEXT_OBS: encoded_observations[player_id],
                    EpisodeKey.ACTION_MASK: action_masks[player_id],
                    EpisodeKey.DONE: dones[player_id],
                    EpisodeKey.REWARD: rewards[player_id]                
                },
                another_player_id:{
                    EpisodeKey.DONE: dones[another_player_id],
                    EpisodeKey.REWARD: rewards[another_player_id]
                }
            }
        if done:
            ret={
                agent_id:{
                    EpisodeKey.DONE: dones[agent_id],
                    EpisodeKey.REWARD: rewards[agent_id]
                } for agent_id in self.agent_ids
            }
        return ret
    
    def get_episode_stats(self):
        return self.stats
    
    def update_episode_stats(self,rewards):
        rewards=[rewards[agent_id] for agent_id in self.agent_ids]
        if len(rewards)==2:
            win=1 if rewards[0]>rewards[1] else 0
            lose=1 if rewards[0]<rewards[1] else 0
            score=1 if win else (0 if lose else 0.5)
            reward=rewards[0]
            self.stats["agent_0"]={
                "win": win,
                "lose": lose,
                "score": score,
                "reward": reward
            }
            
            win=1 if rewards[0]<rewards[1] else 0
            lose=1 if rewards[0]>rewards[1] else 0
            score=1 if win else (0 if lose else 0.5)
            reward=rewards[1]
            self.stats["agent_1"]={
                "win": win,
                "lose": lose,
                "score": score,
                "reward": reward
            }
    
    def is_terminated(self):
        return self.step_data.last()
    
    def encode(self):
        observations=self.step_data.observations
        print(observations)
        player_idx=observations["current_player"]
        player_id="agent_{}".format(player_idx)
        encoded_observations={
            player_id: np.asarray(observations["info_state"][player_idx],dtype=np.float32)
        }
        legal_actions=observations["legal_actions"][player_idx]
        action_mask=np.zeros(self._env.action_spec()["num_actions"],dtype=bool)
        action_mask[legal_actions]=1
        action_masks={
            player_id: action_mask
        }
        return encoded_observations,action_masks
    