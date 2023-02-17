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
        agent2pop = {}
        for pop in populations_cfg:
            if pop['agent_group'] not in agent2pop:
                agent2pop[pop['agent_group']] = [(pop['population_id'], pop['algorithm'])]
            else:
                agent2pop[pop['agent_group']].append((pop['population_id'],pop['algorithm']))

        #TODO: the agent name and population name are confused
        for agent_id in self.agents.training_agent_ids:
            for pop_pair in agent2pop[agent_id]:
                pop_id, algo_cfg = pop_pair
                self.agents[agent_id].add_new_population(
                    pop_id, algo_cfg, self.policy_server
                )

        # generate the first policy
        for agent_id in self.agents.training_agent_ids:
            for population_id in self.agents[agent_id].populations:
                self.gen_new_policy(agent_id, population_id)

        # TODO(jh):Logger
        Logger.warning("after initialization:\n{}".format(self.agents))

    @staticmethod
    def default_agent_id(id):
        return "agent_{}".format(id)

    @staticmethod
    def build_agents(agent_manager_cfg):
        num_agents = agent_manager_cfg['num_agents']
        agent_ids = agent_manager_cfg.get('agent_ids', None)
        share_policies = agent_manager_cfg['share_policies']

        if agent_ids is None:       #default agent_{idx} labelling
            agent_ids = [
                AgentManager.default_agent_id(idx)
                for idx in range(agent_manager_cfg.num_agents)
            ]
        if agent_manager_cfg.share_policies:        #share the agent
            agent = Agent(AgentManager.default_agent_id(0))
            agents = Agents(
                OrderedDict({agent_id: agent for agent_id in agent_ids}), True
            )
        else:
            _agents = [Agent(aid) for aid in agent_ids]

            agents = Agents(
                OrderedDict({agent_id: agent
                             for agent_id, agent in zip(agent_ids, _agents)}),
                share_policies
            )

        return agents

    def gen_new_policy(self, agent_id, population_id):
        # breakpoint()
        policy_id, policy = self.agents[agent_id].gen_new_policy(population_id)
        self.agents[agent_id].add_new_policy(population_id, policy_id)
        self.push_policy_to_remote(agent_id, policy_id, policy)
        # breakpoint()
        return policy_id

    def push_policy_to_remote(self, agent_id, policy_id, policy, version=-1):
        # push to remote
        policy_desc = PolicyDesc(agent_id, policy_id, policy, version)
        ray.get(self.policy_server.push.remote(self.id, policy_desc))
