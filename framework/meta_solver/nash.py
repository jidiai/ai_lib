from .base import MetaSolver
import nashpy as nash
import numpy as np

class Solver(MetaSolver):
    def __init__(self):
        self.iterations=20000
    
    def compute(self, payoff):
        assert len(payoff.shape)==2 and np.all(payoff+payoff.T<1e-6),"only support two-player zero-sum symetric games now.\n payoff:{}".format(payoff)
        eqs=self.compute_nash(payoff)
        return eqs
    
    def compute_nash(self, payoff):
        game=nash.Game(payoff)
        freqs = list(game.fictitious_play(iterations=100000))[-1]  
        eqs = tuple(map(lambda x: x / np.sum(x), freqs))
        return eqs
