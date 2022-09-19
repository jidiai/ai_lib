import numpy as np
import nashpy as nash

class RPP:
    '''
    Relative Population Performance
    '''
    def __init__(self):
        pass
    
    def eval(self,payoff,population_0,population_1):
        '''
        population: {agent_id:[policy_id_idx]}
        perf of pop0 - perf of pop1
        '''
        assert len(population_0)==2 and len(population_1)==2 and np.all(payoff+payoff.T<1e-6),"only support two-player zero-sum symetric games now.\n payoff:{}".format(payoff)
        p0a0_policy_idices=np.array(population_0["agent_0"],dtype=int)
        p0a1_policy_idices=np.array(population_0["agent_1"],dtype=int)
        p1a0_policy_idices=np.array(population_1["agent_0"],dtype=int)
        p1a1_policy_idices=np.array(population_1["agent_1"],dtype=int)
        
        sub_payoff_p0=payoff[p0a0_policy_idices][p1a1_policy_idices]
        sub_payoff_p1=payoff[p1a0_policy_idices][p0a1_policy_idices]
        v_p0=self.compute_nash_value(sub_payoff_p0)
        v_p1=self.compute_nash_value(sub_payoff_p1)
        return v_p0-v_p1
        
    def compute_nash_value(self,payoff):
        game=nash.Game(payoff)
        freqs_A,freqs_B = list(game.fictitious_play(iterations=100000))
        probs_A = tuple(map(lambda x: x / np.sum(x), freqs_A))
        probs_B = tuple(map(lambda x: x / np.sum(x), freqs_B))
        v_A,v_B = game[probs_A,probs_B]
        return v_A