'''
Implement PSRO framework.
'''
from omegaconf import DictConfig

from .. import rollout,agent,training,agent,buffer
from tools.utils.desc.task_desc import TrainingDesc, EvaluationDesc
import ray
import numpy as np

class PSRO:
    def __init__(self,cfg:DictConfig):
        self.cfg=cfg
        self.framework_cfg=self.cfg.framework

        # TODO: add resource limitation
        AgentManager=ray.remote()(agent.AgentManager)
        RolloutManager=ray.remote()(rollout.RolloutManager)
        TrainingManager=ray.remote()(training.TrainingManager)
        DataServer=ray.remote()(buffer.DataServer)
        PolicyServer=ray.remote()(buffer.PolicyServer)

        self.data_server=DataServer.options(
            name="DataServer",
            max_concurrency=200
        ).remote(
            self.cfg.data_server
        )
        
        self.policy_server=PolicyServer.options(
            name="PolicyServer",
            max_concurrency=10
        ).remote(
            self.cfg.policy_server
        )
           
        self.agent_manager=AgentManager.options(
            name="AgentManager",
            max_concurrency=10
        ).remote(
            self.cfg.agent_manager
        )
        
        self.rollout_manager=RolloutManager.options(
            name="RolloutManager",
            max_concurrency=10
        ).remote(
            self.cfg.rollout_manager,
            self.agent_manager.agents
        )
        
        self.training_manager=TrainingManager.options(
            name="TrainingManager",
            max_concurrency=10
        ).remote(
            self.cfg.training_manager
        )
    
    def run(self):        
        ray.get(self.agent_manager.load_initial_policies.remote())
        max_generations=self.framework_cfg.max_generations
        agent_ids=ray.get(self.agent_manager.get_agent_ids.remote())
        for generation in range(max_generations):
            for agent_id in agent_ids:
                self.eval()
                self.train(agent_id)
    
        self.eval()
        
    def train(self,agent_id):
        policy_distributions=ray.get(self.agent_manager.compute_meta_strategy.remote())
        policy_id=ray.get(self.agent_manager.gen_new_policy.remote(agent_id))
        policy_distributions[agent_id]={"policy_ids":np.array([policy_id]),"policy_probs":np.array([1.0],dtype=float)}
        training_desc=TrainingDesc(agent_id,policy_id,policy_distributions)
        training_task_ref=self.training_manager.train.remote(training_desc)
        ray.get(training_task_ref)
        
    def eval(self):
        evaluation_task_ref=self.agent_manager.eval.remote()
        ray.get(evaluation_task_ref)