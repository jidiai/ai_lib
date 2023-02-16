import threading
from agent.agent import Agents
from utils.desc.policy_desc import PolicyDesc
from readerwriterlock import rwlock

from utils.logger import Logger


class PolicyServer:
    """
    TODO(jh) This implementation is still problematic. we should rewrite it in asyncio's way, e.g. should use asyncio.Lock.
    Because there is not yield here, and no resouce contention, no lock is still correct.
    """

    def __init__(self, id, cfg, agents: Agents):
        self.id = id
        self.cfg = cfg
        self.agents = agents
        locks = (
            [rwlock.RWLockWrite()] * len(self.agents)
            if self.agents.share_policies
            else [rwlock.RWLockWrite() for i in range(len(self.agents))]
        )
        self.locks = {
            agent_id: lock for agent_id, lock in zip(self.agents.agent_ids, locks)
        }

        Logger.info("{} initialized".format(self.id))

    def print_agents(self):
        print(f'policy server keys = {self.agents.keys()}')

    async def push(self, caller_id, policy_desc: PolicyDesc):
        # Logger.debug("{} try to push({}) to policy server".format(caller_id,str(policy_desc)))
        agent_id = policy_desc.agent_id
        policy_id = policy_desc.policy_id
        lock = self.locks[agent_id]
        with lock.gen_wlock():
            old_policy_desc = self.agents[agent_id].policy_data.get(policy_id, None)
            if (
                old_policy_desc is None
                or old_policy_desc.version is None
                or old_policy_desc.version < policy_desc.version
            ):
                self.agents[agent_id].policy_data[policy_id] = policy_desc
            else:
                Logger.debug("{}::push() discard order policy".format(self.id))
        # Logger.debug("{} try to push({}) to policy server ends".format(caller_id,str(policy_desc)))

    async def pull(self, caller_id, agent_id, policy_id, old_version=None):
        # Logger.error("{} try to pull({},{},{}) from policy server".format(caller_id,agent_id,policy_id,old_version))
        lock = self.locks[agent_id]
        with lock.gen_rlock():
            if policy_id not in self.agents[agent_id].policy_data:
                ret = None
            else:
                policy_desc: PolicyDesc = self.agents[agent_id].policy_data[policy_id]
                if old_version is None or old_version < policy_desc.version:
                    ret = policy_desc
                else:
                    ret = None
        # Logger.warning("{} try to pull({},{},{}) from policy server ends".format(caller_id,agent_id,policy_id,old_version))
        return ret

    def dump_policy(self):
        pass

    def show_agents(self, agent_id):
        return self.agents[agent_id].policy_data
