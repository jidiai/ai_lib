from registry import registry
from utils.episode import EpisodeKey

import numpy as np
from pettingzoo.classic import connect_four_v3
from ..base_aec_env import BaseAECEnv
from gym.spaces import Box,Discrete

class DefaultFeatureEncoder:
    def encode(self,observation,agent_id):
        idx=int(agent_id[-1])
        obs=observation["observation"]
        non_empty=obs[:,:,:1]+obs[:,:,1:]
        empty=1-non_empty
        obs=np.concatenate([obs,non_empty,empty],axis=-1)
        obs=obs.flatten()
        feature=np.zeros(len(obs)+1,dtype=np.float32)
        feature[0]=idx
        feature[1:]=obs
        action_mask=observation["action_mask"]
        return feature,action_mask
    
    @property
    def observation_space(self):
        return Box(0.0,1.0,shape=(6*7*4+1,))
    
    @property
    def action_space(self):
        return Discrete(7)

@registry.registered(registry.ENV,"connect_four")
class ConnectFourEnv(BaseAECEnv):
    '''
    https://www.pettingzoo.ml/classic/connect_four
    action_space: Discrete(7)
    observation_space: Box(6,7,2) Value{0,1}
    '''
    def __init__(self,id,seed,cfg):
        self.id=id
        self.seed=seed
        self.cfg=cfg

        self._env=connect_four_v3.env()        
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
        observation_all, reward, done, _ = self._env.last() 
        
        self._last_observation=observation_all["observation"]
        
        observation,action_mask=self.feature_encoders[agent_id].encode(observation_all,agent_id)
        
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
        
        self._is_terminated=np.all(list(self._env.dones.values()))
        
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
        assert mode=="cmd"
        agent_id=self._last_agent_id
        #print(agent_id)
        # NOTE .last() is actually the next player's
        piece_0=1 if self.id_mapping(self._env.agent_selection)=="agent_0" else 2
        piece_1=3-piece_0
        observation_all,reward,done,_=self._env.last()
        observation=observation_all["observation"]
        # reconstruct board
        board=observation[:,:,0]*piece_0+observation[:,:,1]*piece_1
        if self.step_ctr==0:
            print("[step {}] board:\n{}\n".format(self.step_ctr,board))
        else:
            print("[step {}] {}'s action:{} reward:{} board:\n{}\n".format(self.step_ctr,agent_id,self._last_action,reward,board))
        