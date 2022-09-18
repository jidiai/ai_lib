from collections import OrderedDict
from tools.utils.desc.policy_desc import PolicyDesc
from . import PolicyFactory
from tools.utils.logger import Logger
import ray

class Agents(OrderedDict):
    def __init__(self,share_policies,*args,**kwargs):
        self.share_policies=share_policies
        super().__init__(*args,**kwargs)


class Agent:
    def __init__(self,id,algorithm_cfg,policy_server):
        self.id=id
        self.policy_ids=[] # used in agent_manager
        self.policy_id2idx={}
        self.policy_data={} # used in rollout_worker
        self.policy_factory=PolicyFactory(self.id,algorithm_cfg,policy_server)
        self.policy_server=policy_server

    def gen_new_policy(self):
        policy_id,policy=self.policy_factory.gen_new_policy()
        self.add_new_policy(policy_id,policy)
        return policy_id

    def add_new_policy(self,policy_id,policy):
        if policy_id in self.policy_id2idx:
            Logger.error("Cannot add {}, which is already in the policy pool")
            return
        self.policy_ids.append(policy_id)
        self.policy_id2idx[policy_id]=len(self.policy_id2idx)
        self.policy_data[policy_id]={}

        # push to remote
        policy_desc=PolicyDesc(
            self.id,
            policy_id,
            policy
        )
        ray.get(self.policy_server.push.remote(policy_desc))