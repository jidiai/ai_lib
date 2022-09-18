from email import policy
from tools.utils.desc.policy_desc import PolicyDesc
from readerwriterlock import rwlock

class PolicyServer:
    def __init__(self,agent_ids):
        self.policies={agent_id:{} for agent_id in agent_ids}
        self.locks={agent_id:rwlock.RWLockWrite() for agent_id in agent_ids}
        
    def push(self,policy_desc:PolicyDesc):
        agent_id=policy_desc.agent_id
        policy_id=policy_desc.policy_id
        policy=policy_desc.policy  
        # TODO(jh): check version is updated
        # version=policy_desc.version
        lock=self.locks[agent_id]
        with lock.gen_wlock():
            self.policies[agent_id][policy_id]=policy
        
    def pull(self,agent_id,policy_id,old_version=None):
        lock=self.locks[agent_id]
        with lock.gen_rlock():
            if policy_id not in self.policies[agent_id]:
                return None     
            policy_desc:PolicyDesc=self.policies[agent_id][policy_id]
            if old_version is None or old_version<policy_desc.version:
                return policy_desc
            else:
                return None
        
    def dump_policy(self):
        pass