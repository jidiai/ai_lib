from utils.episode import EpisodeKey
from ..base_env import BaseEnv

try:
    from gfootball import env as gfootball_official_env
    from gfootball.env.football_env import FootballEnv
except ImportError as e:
    raise e(
        "Please install Google football evironment before use: https://github.com/google-research/football"
    ) from None

from .state import State
import numpy as np
from .rewarder_basic import Rewarder
from .stats_basic import StatsCaculator

from utils.timer import global_timer
from utils.logger import Logger

class GRFootballEnv(BaseEnv):
    def __init__(self,id,seed,cfg):
        super().__init__(id,seed)
        self.cfg=cfg
        scenario_config=self.cfg["scenario_config"]
        scenario_name=scenario_config["env_name"]
        assert scenario_name in ["5_vs_5","10_vs_10_kaggle"],"Because of some bugs in envs, only these scenarios are supported now. See README"
        # scenario_config["other_config_options"]["game_engine_random_seed"]=int(seed)
        self._env:FootballEnv=gfootball_official_env.create_environment(**scenario_config)
        self.agent_ids=["agent_0","agent_1"]
        self.num_players={
            "agent_0":scenario_config["number_of_left_players_agent_controls"],
            "agent_1":scenario_config["number_of_right_players_agent_controls"]
        }
        self.slices={
            "agent_0": slice(0,self.num_players["agent_0"]),
            "agent_1": slice(self.num_players["agent_0"],None)
        }
        for num in self.num_players.values():
            assert num>0,"jh: if built-in ai is wanted, use built_in model."
        self.feature_encoders={
            "agent_0": None,
            "agent_1": None
        }
        self.num_actions=19
        self.rewarder=Rewarder(self.cfg.reward_config)
        self.stats_calculators={
            "agent_0": StatsCaculator(),
            "agent_1": StatsCaculator(),
        }
    
    @property
    def num_players_total(self):
        return sum(self.num_players.values())
    
    def reset(self,custom_reset_config):
        self.feature_encoders=custom_reset_config["feature_encoders"]
        self.main_agent_id=custom_reset_config["main_agent_id"]
        self.rollout_length=custom_reset_config["rollout_length"]
        
        self.step_ctr=0
        
        global_timer.record("env_step_start")
        observations=self._env.reset()
        global_timer.time("env_step_start","env_step_end","env_step")
        
        self.states=[State() for i in range(self.num_players_total)]
        for stats_calculator in self.stats_calculators.values():
            stats_calculator.reset()
        
        assert len(observations)==len(self.states)
        for o,s in zip(observations,self.states):
            s.update_obs(o)
        
        encoded_observations,action_masks=self.encode()
        dones={
            k: np.zeros((v,1),dtype=bool) 
            for k,v in self.num_players.items()
        }
        
        rets={
            agent_id: {
                EpisodeKey.NEXT_OBS: encoded_observations[agent_id],
                EpisodeKey.ACTION_MASK: action_masks[agent_id],
                EpisodeKey.DONE: dones[agent_id]
            } for agent_id in self.agent_ids
        }
        return rets
    
    def step(self,actions):
        self.step_ctr+=1
        
        actions=np.concatenate([actions[agent_id][EpisodeKey.ACTION] for agent_id in self.agent_ids],axis=0).flatten()
        
        global_timer.record("env_core_step_start")
        observations,rewards,done,info=self._env.step(actions)
        global_timer.time("env_core_step_start","env_core_step_end","env_core_step")
        
        assert len(observations)==len(self.states) and len(actions)==len(self.states)
        for o,a,s in zip(observations,actions,self.states):
            s.update_action(a)
            s.update_obs(o)
    
        global_timer.record("reward_start")
        rewards=self.get_reward(rewards)
        global_timer.time("reward_start","reward_end","reward")
        
        global_timer.record("stats_start")
        self.update_episode_stats(rewards)
        global_timer.time("stats_start","stats_end","stats")
        
        global_timer.record("feature_start")
        encoded_observations,action_masks=self.encode()
        global_timer.time("feature_start","feature_end","feature")
        
        # if np.any(np.array(s.obs["score"])-np.array(s.prev_obs["score"])):
        #     Logger.error("score_reward: {} | {} | {}".format(info["score_reward"],s.obs["score"],s.prev_obs["score"]))
        #     assert info["score_reward"]
        #     done=True
        
        if info["score_reward"]:
            #Logger.error("score diff: {}".format(np.any(np.array(s.obs["score"])-np.array(s.prev_obs["score"]))))
            done=True
           
        dones={
            k: np.full((v,1),fill_value=done,dtype=bool) 
            for k,v in self.num_players.items()
        }
                 
        rets={
                agent_id: {
                    EpisodeKey.NEXT_OBS: encoded_observations[agent_id],
                    EpisodeKey.ACTION_MASK: action_masks[agent_id],
                    EpisodeKey.REWARD: rewards[agent_id],
                    EpisodeKey.DONE: dones[agent_id]
                } for agent_id in self.agent_ids

        }
        return rets
    
    def is_terminated(self):
        return self.step_ctr>=self.rollout_length
    
    def split(self,arr):
        ret={
            agent_id:arr[self.slices[agent_id]]
            for agent_id in self.agent_ids
        }        
        return ret
    
    def get_reward(self,rewards):
        rewards = np.array([
            [self.rewarder.calc_reward(reward, state)]
                for reward,state in zip(rewards,self.states)
        ],dtype=float)

        rewards = self.split(rewards)
        return rewards
        
    def encode(self):
        states=self.split(self.states)
        encoded_observations={
            agent_id:np.array(self.feature_encoders[agent_id].encode(states[agent_id]),dtype=np.float32)
            for agent_id in self.agent_ids
        }
        action_masks={
            agent_id:encoded_observations[agent_id][...,:self.num_actions] 
            for agent_id in self.agent_ids
        }
        return encoded_observations,action_masks
    
    def update_episode_stats(self,rewards):
        '''
        we only count statistics for main agent now
        '''
        states=self.split(self.states)
        for agent_id in self.agent_ids:
            for idx,state in enumerate(states[agent_id]):
                self.stats_calculators[agent_id].calc_stats(state,rewards[agent_id][idx][0],idx)
            
    def get_episode_stats(self):
        return {
            agent_id:self.stats_calculators[agent_id].stats
            for agent_id in self.agent_ids
        }
        
    def render(self,mode="human"):
        assert mode=="human"
        self._env.render()