from .base import MetaSolver
import numpy as np

class Solver(MetaSolver):
    def __init__(self):
        self.iterations=20000
    
    def compute(self, payoff):
        assert len(payoff.shape)==2 and np.all(payoff+payoff.T<1e-6),"only support two-player zero-sum symetric games now.\n payoff:{}".format(payoff)
        eqs=(
            np.full(payoff.shape[0],fill_value=1/payoff.shape[0],dtype=np.float32),
            np.full(payoff.shape[0],fill_value=1/payoff.shape[1],dtype=np.float32),
        )
        return eqs

