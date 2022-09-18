from typing import Dict, OrderedDict

from tools.utils.desc.policy_desc import PolicyDesc
from tools.envs.env_factory import make_envs
from .rollout_func import rollout_func
import numpy as np
from ..agent import Agent
from ..utils.random import set_random_seed
from tools.utils.remote import get_actor
import ray
from tools.utils.desc.task_desc import RolloutDesc

class RolloutWorker:
    def __init__(self,id,seed,cfg,agents:Dict[str,Agent]):
        self.id=id
        self.seed=seed
        set_random_seed(self.seed)

        self.cfg=cfg
        self.agents=agents
        self.policy_server=get_actor(self.id,"PolicyServer")
        self.data_server=get_actor(self.id,"DataServer")
        self.envs:Dict=make_envs(self.id,self.seed,self.cfg.envs)

    def rollout(self,rollout_desc:RolloutDesc,eval=False,rollout_epoch=None):
        assert len(self.envs)==1,"jh: currently only single env is supported"
        env=list(self.envs.values)[0]

        policy_distributions=rollout_desc.policy_distributions
        if not eval:
            policy_ids=self.sample_policies(policy_distributions)
        else:
            policy_ids={}
            for agent_id,policy_id in policy_distributions.items():
                assert isinstance(policy_id,str)
                policy_ids[agent_id]=policy_id

        # pull policies from remote
        self.pull_policies(policy_ids)
        behaving_policies=self.get_policies(policy_ids)

        result=rollout_func(
            eval,
            self,
            rollout_desc,
            env,
            behaving_policies,
            self.data_server,
            rollout_length=self.cfg.rollout_length,
            sample_length=self.cfg.sample_length
            # rollout_epoch,
            # decaying_exploration_cfg=self.cfg.decaying_exploration
        )

        return result

    def get_policies(self,policy_ids):
        policies=OrderedDict()
        for agent_id,policy_id in policy_ids.items():
            policy=self.agents[agent_id].policy_data[policy_id]["desc"].policy
            policies[agent_id]=(policy_id,policy)
        return policies

    def pull_policies(self,policy_ids):
        for agent_id,policy_id in policy_ids.items():
            if policy_id not in self.agents[agent_id].policy_data:
                policy_desc=ray.get(self.policy_server.pull.remote(agent_id,policy_id,version=None))
                self.agents[agent_id].policy_data[policy_id]={"desc":policy_desc}
            else:
                old_policy_desc:PolicyDesc=self.agents[agent_id].policy_data[policy_id]["desc"]
                policy_desc=ray.get(self.policy_server.pull.remote(agent_id,policy_id,version=old_policy_desc.version))
                if policy_desc is not None:
                    self.agents[agent_id].policy_data[policy_id]["desc"]=policy_desc

    def sample_policies(self,policy_distributions):
        policy_ids=OrderedDict()
        for agent_id,distribution in policy_distributions.items():
            policy_ids[agent_id]=self.sample_policy(distribution)
        return policy_ids

    def sample_policy(self,policy_distribution):
        policy_ids=policy_distribution["policy_ids"]
        policy_probs=policy_distribution["policy_probs"]

        policy_probs=np.array(policy_probs)
        policy_probs=policy_probs/np.sum(policy_probs)
        policy_id=np.random.choice(a=policy_ids,p=policy_probs)
        return policy_id
