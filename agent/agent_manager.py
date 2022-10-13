from typing import OrderedDict
from utils.desc.policy_desc import PolicyDesc
from utils.distributed import get_actor
from .agent import Agent, Agents
from algorithm.mappo.policy import MAPPO
from utils.logger import Logger
import ray
from algorithm.q_learning.policy import QLearning


class AgentManager:
    def __init__(self, cfg):
        self.id = "AgentManager"
        self.cfg = cfg
        self.policy_server = get_actor("AgentManager", "PolicyServer")
        self.agents = self.build_agents(cfg)
        Logger.info("AgentManager initialized")

    def load_policy(self, agent_id, population_id, policy_id, policy_dir):
        # TODO(jh)
        policy = MAPPO.load(policy_dir, env_agent_id=agent_id)
        agent = self.agents[agent_id]
        agent.add_new_policy(population_id, policy_id)
        self.push_policy_to_remote(agent_id, policy_id, policy)
        return policy_id, policy

    def get_agent_ids(self):
        return [agent.id for agent in self.agents]

    def eval(self):
        return self.evaluation_manager.eval()

    def initialize(self, populations_cfg):
        # add populations
        for agent_id in self.agents.training_agent_ids:
            # print('training agent id =', agent_id)
            for population_cfg in populations_cfg:
                # print('population cfg = ', population_cfg)
                population_id = population_cfg["population_id"]
                algorithm_cfg = population_cfg["algorithm"]
                self.agents[agent_id].add_new_population(population_id, algorithm_cfg, self.policy_server)
        # print('self.agents = ', self.agents[agent_id])

        for population_cfg in populations_cfg:
            population_id = population_cfg["population_id"]
            algorithm_cfg = population_cfg.algorithm
            policy_init_cfg = algorithm_cfg.get('policy_init_cfg', None)
            if policy_init_cfg is None:
                continue
            for agent_id, agent_policy_init_cfg in policy_init_cfg.items():
                print('agent id = ', agent_id)
                print('init cfg = ', agent_policy_init_cfg)
                agent_initial_policies = agent_policy_init_cfg.get("initial_policies", None)
                if agent_initial_policies is None:
                    continue
                for policy_cfg in agent_initial_policies:
                    policy_id = policy_cfg["policy_id"]
                    policy_dir = policy_cfg["policy_dir"]
                    policy = QLearning.load(policy_dir)
                    agent = self.agents[agent_id]
                    agent.add_new_policy(population_id, policy_id)
                    self.push_policy_to_remote(agent_id, policy_id, policy)

        # print('self.agents =', self.agents.agent_ids)
        # print('policy server agent_1 = ', ray.get(self.policy_server.show_agents.remote(agent_id = 'agent_1')))
        # print('policy server agent_0 = ', ray.get(self.policy_server.show_agents.remote(agent_id = 'agent_0')))
        #
        # raise NotImplementedError

        # raise NotImplementedError
        #
        # for population_cfg in populations_cfg:
        #     population_id=population_cfg["population_id"]
        #     algorithm_cfg=population_cfg.algorithm
        #     policy_init_cfg=algorithm_cfg.get("policy_init_cfg",None)
        #     if policy_init_cfg is None:
        #         continue
        #     for agent_id,agent_policy_init_cfg in policy_init_cfg.items():
        #         agent_initial_policies=agent_policy_init_cfg.get("initial_policies",None)
        #         if agent_initial_policies is None:
        #             continue
        #         for policy_cfg in agent_initial_policies:
        #             policy_id=policy_cfg["policy_id"]
        #             policy_dir=policy_cfg["policy_dir"]
        #             self.load_policy(agent_id,population_id,policy_id,policy_dir)
        #             Logger.info(f"Load initial policy {policy_id} from {policy_dir}")

        # generate the first policy
        for agent_id in self.agents.training_agent_ids:
            for population_id in self.agents[agent_id].populations:
                self.gen_new_policy(agent_id, population_id)

        # print('self.agents =', self.agents.agent_ids)
        # print('policy server agent_1 = ', ray.get(self.policy_server.show_agents.remote(agent_id = 'agent_1')))
        # print('policy server agent_0 = ', ray.get(self.policy_server.show_agents.remote(agent_id = 'agent_0')))
        #
        # raise NotImplementedError

        # TODO(jh):Logger
        Logger.warning(
            "after initialization:\n{}".format(self.agents)
        )

    @staticmethod
    def default_agent_id(id):
        return "agent_{}".format(id)

    @staticmethod
    def build_agents(agent_manager_cfg):
        agent_ids = [AgentManager.default_agent_id(idx) for idx in range(agent_manager_cfg.num_agents)]
        if agent_manager_cfg.share_policies:
            agent = Agent(AgentManager.default_agent_id(0))
            agents = Agents(OrderedDict({agent_id: agent for agent_id in agent_ids}), True)
        else:
            agents = [Agent(AgentManager.default_agent_id(idx)) for idx in range(len(agent_ids))]
            agents = Agents(OrderedDict({agent_id: agent for agent_id, agent in zip(agent_ids, agents)}), False)
        return agents

    def gen_new_policy(self, agent_id, population_id):
        policy_id, policy = self.agents[agent_id].gen_new_policy(population_id)
        self.agents[agent_id].add_new_policy(population_id, policy_id)
        self.push_policy_to_remote(agent_id, policy_id, policy)
        return policy_id

    def push_policy_to_remote(self, agent_id, policy_id, policy, version=-1):
        # push to remote
        policy_desc = PolicyDesc(
            agent_id,
            policy_id,
            policy,
            version
        )
        ray.get(self.policy_server.push.remote(self.id, policy_desc))
