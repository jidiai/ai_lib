from gym.spaces import Box,Discrete,Space
import pyspiel
from open_spiel.python.algorithms import exploitability
import copy
import numpy as np
from pettingzoo.utils import agent_selector, wrappers

class LeducPokerRawEnv:
    
    metadata = {"render_modes": ["human"], "name": "leduc_poker_open_spiel"}
    
    def __init__(self,seed):
        # so we actually use np.random to simulate env moves
        self.seed(seed)
        
        self._game=pyspiel.load_game("leduc_poker",)
        self.state=None
        self.dones=None        
        self.curr_player_idx=None
        self.possible_agents = ["player_{}".format(r) for r in range(2)]  
 
    def seed(self,seed):
        self._seed=seed
        np.random.seed(seed)
        
    def render(self,mode="human"):
        txt=self.state.observation_string(self.curr_player_idx)
        txt=txt.replace("Private: 0","Private: J1")
        txt=txt.replace("Private: 1","Private: Q1")
        txt=txt.replace("Private: 2","Private: K1")
        txt=txt.replace("Private: 3","Private: J2")
        txt=txt.replace("Private: 4","Private: Q2")
        txt=txt.replace("Private: 5","Private: K2")
        txt=txt.replace("Public: 0","Public: J1")
        txt=txt.replace("Public: 1","Public: Q1")
        txt=txt.replace("Public: 2","Public: K1")
        txt=txt.replace("Public: 3","Public: J2")
        txt=txt.replace("Public: 4","Public: Q2")
        txt=txt.replace("Public: 5","Public: K2")
        print(txt+"\n")
        
    def agent_iter(self):
        while not np.all(self.dones):
            yield "player_{}".format(self.curr_player_idx)
        # for getting the reward
        yield "player_{}".format(self.curr_player_idx)
        yield "player_{}".format(self.curr_player_idx)
        
    def _sample(self,actions_and_probs):
        actions, probs = zip(*actions_and_probs)
        return np.random.choice(actions, p=probs)
    
    def _skip(self):
        while self.state.is_chance_node():
            p = self.state.chance_outcomes()
            a = self._sample(p)
            self.state.apply_action(a)

    def reset(self):
        self.state=self._game.new_initial_state()
        self.dones=[False,False]
        self._skip()
        self.curr_player_idx=self.state.current_player()
    
    def step(self,action):        
        if not self.state.is_terminal():
            self.state.apply_action(action)
            self._skip()
            if not self.state.is_terminal():
                self.curr_player_idx=self.state.current_player()
            else:
                self.dones=[True,True]
                self.curr_player_idx=1-self.curr_player_idx
        else:
            assert action is None
            self.curr_player_idx=1-self.curr_player_idx
    
    def last(self):
        done=self.state.is_terminal()
        reward=self.state.returns()[self.curr_player_idx]
        info={}
        return copy.deepcopy(self.state),reward,done,info