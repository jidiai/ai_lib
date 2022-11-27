"""
Implement SARL/MARL
"""
import traceback

import rollout, training, buffer
from agent import AgentManager
from evaluation.evaluation_manager import EvaluationManager
from agent.policy_data.policy_data_manager import PolicyDataManager
from framework.scheduler.marl_scheduler import MARLScheduler
from utils.distributed import get_resources
from utils.logger import Logger

import ray


class MARLRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.framework_cfg = self.cfg.framework
        self.id = self.framework_cfg.name

        RolloutManager = ray.remote(
            **get_resources(cfg.rollout_manager.distributed.resources)
        )(rollout.RolloutManager)
        TrainingManager = ray.remote(
            **get_resources(cfg.training_manager.distributed.resources)
        )(training.TrainingManager)
        DataServer = ray.remote(**get_resources(cfg.data_server.distributed.resources))(
            buffer.DataServer
        )
        PolicyServer = ray.remote(
            **get_resources(cfg.policy_server.distributed.resources)
        )(buffer.PolicyServer)

        agents = AgentManager.build_agents(self.cfg.agent_manager)

        self.data_server = DataServer.options(
            name="DataServer", max_concurrency=500
        ).remote("DataServer", self.cfg.data_server)

        self.policy_server = PolicyServer.options(
            name="PolicyServer", max_concurrency=500
        ).remote("PolicyServer", self.cfg.policy_server, agents)

        self.rollout_manager = RolloutManager.options(
            name="RolloutManager", max_concurrency=500
        ).remote("RolloutManager", self.cfg.rollout_manager, agents)

        self.training_manager = TrainingManager.options(
            name="TrainingManager", max_concurrency=50
        ).remote("TrainingManager", self.cfg.training_manager)

        self.agent_manager = AgentManager(self.cfg.agent_manager)
        self.policy_data_manager = PolicyDataManager(
            self.cfg.policy_data_manager, self.agent_manager
        )
        self.evaluation_manager = EvaluationManager(
            self.cfg.evaluation_manager, self.agent_manager, self.policy_data_manager
        )

        if self.id == "marl":
            self.scheduler = MARLScheduler(
                self.cfg.framework,
                self.agent_manager,
                "default",
                self.policy_data_manager,
            )
        else:
            raise NotImplementedError

    def run(self):
        self.agent_manager.initialize(self.cfg.populations)
        if self.cfg.eval_only:
            self.evaluation_manager.eval()
        else:
            while True:
                self.evaluation_manager.eval(framework="marl")
                training_desc = self.scheduler.get_task()
                if training_desc is None:
                    self.evaluation_manager.eval(framework="psro")
                    break
                Logger.info("training_desc: {}".format(training_desc))
                try:
                    training_task_ref = self.training_manager.train.remote(
                        training_desc
                    )
                except:
                    print(traceback.format_exc())
                ray.get(training_task_ref)
                self.scheduler.submit_result(None)
            Logger.info("MARLRunning {} ended".format(self.id))

    def close(self):
        ray.get(self.training_manager.close.remote())
        ray.get(self.rollout_manager.close.remote())
