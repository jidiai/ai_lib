'''
Implement PSRO framework.
'''
from collections import OrderedDict

import rollout,agent,training,agent,buffer
from agent import AgentManager
from evaluation.evaluation_manager import EvaluationManager
from agent.policy_data.policy_data_manager import PolicyDataManager
from framework.scheduler.psro_scheduler import PSROScheduler
import ray
import numpy as np
from utils.distributed import get_resources
from utils.logger import Logger

class PBTRunner:
    def __init__(self,cfg):
        self.cfg=cfg
        self.framework_cfg=self.cfg.framework
        self.id=self.framework_cfg.name

        ###### Initialize Components #####

        RolloutManager=ray.remote(**get_resources(cfg.rollout_manager.distributed.resources))(rollout.RolloutManager)
        TrainingManager=ray.remote(**get_resources(cfg.training_manager.distributed.resources))(training.TrainingManager)
        DataServer=ray.remote(**get_resources(cfg.data_server.distributed.resources))(buffer.DataServer)
        PolicyServer=ray.remote(**get_resources(cfg.policy_server.distributed.resources))(buffer.PolicyServer)

        # the order of creation is important? cannot have circle reference
        # create agents
        agents=AgentManager.build_agents(self.cfg.agent_manager)

        self.data_server=DataServer.options(
            name="DataServer",
            max_concurrency=500
        ).remote(
            "DataServer",
            self.cfg.data_server
        )
        
        self.policy_server=PolicyServer.options(
            name="PolicyServer",
            max_concurrency=500
        ).remote(
            "PolicyServer",
            self.cfg.policy_server,
            agents
        )
        
        self.rollout_manager=RolloutManager.options(
            name="RolloutManager",
            max_concurrency=500
        ).remote(
            "RolloutManager",
            self.cfg.rollout_manager,
            agents
        )
        
        self.training_manager=TrainingManager.options(
            name="TrainingManager",
            max_concurrency=50
        ).remote(
            "TrainingManager",
            self.cfg.training_manager
        )

        # NOTE: self.agents is not shared with remote actors.
        self.agent_manager=AgentManager(self.cfg.agent_manager)
        self.policy_data_manager=PolicyDataManager(self.cfg.policy_data_manager,self.agent_manager)
        self.evaluation_manager=EvaluationManager(self.cfg.evaluation_manager,self.agent_manager,self.policy_data_manager)
        
        # TODO(jh): scheduler is designed for future distributed purposes.
        if self.id=="psro":
            # check there is only one default population.
            population_cfgs=self.cfg.populations
            assert len(population_cfgs)==1 and population_cfgs[0]["population_id"]=="default"
            self.scheduler=PSROScheduler(self.cfg.framework,self.agent_manager,"default",self.policy_data_manager)
        else:
            raise NotImplementedError
        
        Logger.info("PBTRunner {} initialized".format(self.id))
        
    def run(self): 
        self.agent_manager.initialize(self.cfg.populations)
        if self.cfg.eval_only:
            self.evaluation_manager.eval()
        else: 
            while True:
                self.evaluation_manager.eval()
                training_desc=self.scheduler.get_task()
                if training_desc is None:
                    break
                Logger.info("training_desc: {}".format(training_desc))
                training_task_ref=self.training_manager.train.remote(training_desc)
                ray.get(training_task_ref)
                self.scheduler.submit_result(None)
            Logger.info("PBTRunner {} ended".format(self.id))
        
    def close(self):
        ray.get(self.training_manager.close.remote())
        ray.get(self.rollout_manager.close.remote())