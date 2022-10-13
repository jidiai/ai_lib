from typing import OrderedDict
from registry import registry
from agent.agent_manager import AgentManager

from agent.policy_data.policy_data_manager import PolicyDataManager
from utils.logger import Logger
from agent import Population
from utils.desc.task_desc import TrainingDesc
import numpy as np
import importlib

class PSROScheduler:
    '''
    TODO(jh): abstract it later
    '''
    def __init__(self,cfg,agent_manager:AgentManager,population_id:str,policy_data_manager:PolicyDataManager):
        self.cfg=cfg
        self.agent_manager=agent_manager
        self.agents=self.agent_manager.agents
        self.population_id=population_id
        self.policy_data_manager=policy_data_manager
        self.meta_solver_type=self.cfg.get("meta_solver","nash")
        self.sync_training=self.cfg.get("sync_training",False)

        Logger.warning("use meta solver type: {}".format(self.meta_solver_type))
        solver_module=importlib.import_module("framework.meta_solver.{}".format(self.meta_solver_type))
        self.meta_solver=solver_module.Solver()
        self._schedule=self._gen_schedule()
    
    def _gen_schedule(self):
        max_generations=self.cfg.max_generations
        for generation_ctr in range(max_generations):
            for training_agent_id in self.agents.training_agent_ids:
                # get all available policy_ids from the population
                agent_id2policy_ids=OrderedDict()
                agent_id2policy_indices=OrderedDict()
                for agent_id in self.agents.keys():
                    population:Population=self.agents[agent_id].populations[self.population_id]
                    agent_id2policy_ids[agent_id]=population.policy_ids
                    agent_id2policy_indices[agent_id]=np.array([self.agents[agent_id].policy_id2idx[policy_id] for policy_id in population.policy_ids])
                    
                # get payoff matrix
                payoff_matrix=self.policy_data_manager.get_matrix_data("payoff",agent_id2policy_indices)
                    
                # compute nash
                equlibrium_distributions=self.meta_solver.compute(payoff_matrix)
                
                policy_distributions={}
                for probs,(agent_id,policy_ids) in zip(equlibrium_distributions,agent_id2policy_ids.items()):
                    policy_distributions[agent_id]=OrderedDict(zip(policy_ids,probs))
                
                # gen new policy
                training_policy_id=self.agent_manager.gen_new_policy(agent_id,self.population_id)
                policy_distributions[training_agent_id]={training_policy_id:1.0}

                Logger.warning("********** Generation[{}] Agent[{}] START **********".format(generation_ctr,training_agent_id))
                
                stopper=registry.get(registry.STOPPER,self.cfg.stopper.type)(policy_data_manager=self.policy_data_manager,**self.cfg.stopper.kwargs)
                
                training_desc=TrainingDesc(
                    training_agent_id,
                    training_policy_id,
                    policy_distributions,
                    self.agents.share_policies,
                    self.sync_training,
                    stopper
                )
                yield training_desc
            
    def get_task(self):
        try:
            task=next(self._schedule)
            return task
        except StopIteration:
            return None
    
    def submit_result(self,result):
        pass