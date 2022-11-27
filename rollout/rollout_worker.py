import copy
from typing import Dict

from collections import OrderedDict
from utils.desc.policy_desc import PolicyDesc
from envs.env_factory import make_envs
import importlib
import numpy as np
from agent.agent import Agent, Agents
from utils.distributed import get_actor
import ray
from utils.desc.task_desc import RolloutDesc
from utils.timer import global_timer
import random
import torch
from utils.episode import EpisodeKey
from utils.random import set_random_seed


def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        arr = torch.FloatTensor(arr)
    return arr


class Q_mixing_wrapper:
    def __init__(self, policy_distribution):
        self.policy_distribution = policy_distribution
        self.policy_pool = OrderedDict()
        self._feature_encoder = None
        self.mode = 'epsilon_greedy'  # [greedy, epsilon_greedy, softmax]
        self.epsilon = 0.1
        self.temperature = 1

    def register_policy(self, policy):
        for pid, policy in policy.items():
            self.policy_pool[pid] = policy

    @property
    def feature_encoder(self):
        self._feature_encoder = list(self.policy_pool.values())[0].feature_encoder
        return self._feature_encoder

    def get_initial_state(self, batch_size):
        rnn_state = list(self.policy_pool.values())[0].get_initial_state(batch_size=batch_size)
        return rnn_state

    def compute_action(self, **kwargs):
        mix_q = None
        mix_action_mask = None
        for pid, probs in self.policy_distribution.items():
            policy = self.policy_pool[pid]
            policy_rets = policy.value_function(**kwargs)
            if mix_q is None:
                mix_q = policy_rets[EpisodeKey.STATE_ACTION_VALUE] * probs
                mix_action_mask = policy_rets[EpisodeKey.ACTION_MASK]
            else:
                mix_q = mix_q + policy_rets[EpisodeKey.STATE_ACTION_VALUE] * probs
                mix_action_mask += policy_rets[EpisodeKey.ACTION_MASK]

        mix_action_mask = mix_action_mask / (mix_action_mask + 1e-10)
        return self.act(mix_q, mix_action_mask, self.epsilon, self.temperature)

    def act(self, q, action_mask, epsilon, temperature):
        q = to_tensor(q)
        action_mask = to_tensor(action_mask)
        if self.mode == 'greedy':
            actions = torch.argmax(q, dim=-1, keepdim=True)
            action_probs = torch.zeros_like(q)
            action_probs[torch.arange(action_probs.shape[0], device=action_probs.device), actions[:, 0]] = 1.0
        elif self.mode == 'epsilon_greedy':
            best_actions = torch.argmax(q, dim=-1, keepdim=True)
            action_probs = action_mask / torch.sum(action_mask, dim=-1, keepdim=True) * epsilon
            action_probs[torch.arange(action_probs.shape[0], device=action_probs.device), best_actions[:, 0]] += (
                        1 - epsilon)
            action_probs = action_probs / torch.sum(action_probs, dim=-1, keepdims=True)
            actions = torch.multinomial(action_probs, num_samples=1)
        elif self.mode == 'softmax':
            q = q * temperature
            q = action_mask * q + (1 - action_mask) * (-10e9)
            action_probs = torch.softmax(q, dim=-1)
            actions = torch.multinomial(action_probs, num_samples=1)
        else:
            raise NotImplementedError
        return {EpisodeKey.ACTION: actions, EpisodeKey.ACTION_PROBS: action_probs}


class RolloutWorker:
    def __init__(self, id, seed, cfg, agents: Agents):
        self.id = id
        self.seed = seed
        # set_random_seed(self.seed)

        self.cfg = cfg
        self.agents = agents
        self.policy_server = get_actor(self.id, "PolicyServer")
        self.data_server = get_actor(self.id, "DataServer")
        self.envs: Dict = make_envs(self.id, self.seed, self.cfg.envs)

        module = importlib.import_module("rollout.{}".format(self.cfg.rollout_func_name))
        self.rollout_func = module.rollout_func

        self.mix_opponent = cfg.get('mix_opponent', False)
        self.policy_distribution_record = []

    def rollout(self, rollout_desc: RolloutDesc, eval=False, rollout_epoch=0):
        global_timer.record("rollout_start")
        assert len(self.envs) == 1, "jh: currently only single env is supported"
        env = list(self.envs.values())[0]

        if rollout_desc.type == 'evaluation':
            mix_oppo = False
            assert eval, print('eval= ', eval)
        elif rollout_desc.type == 'rollout':
            mix_oppo = self.mix_opponent
            assert not eval, print('eval= ', eval)
        # print(f'type = {rollout_desc.type}, rolloutDesc = {rollout_desc}')

        if self.agents.share_policies:
            # make order-invariant
            rollout_desc = self.random_permute(rollout_desc)

        policy_distributions = rollout_desc.policy_distributions
        if mix_oppo:
            global_timer.time("rollout_start", "sample_end", "sample")

            behaving_policies = self.get_mix_policies(policy_distributions)
            global_timer.time("sample_end", "policy_update_end", "policy_update")
            # print('policy dist = ', policy_distributions)
            # print('behaving pid = ', behaving_policies)


        else:
            policy_ids = self.sample_policies(policy_distributions)
            # print(f'policy distribution = {policy_distributions}, polict uds = {policy_ids}')

            # print('policy dist = ', policy_distributions)
            # print('policy ids = ', policy_ids)
            global_timer.time("rollout_start", "sample_end", "sample")
            # pull policies from remote
            self.pull_policies(policy_ids)
            behaving_policies = self.get_policies(policy_ids)
            # print('behaving pid = ', behaving_policies)
            global_timer.time("sample_end", "policy_update_end", "policy_update")

        rollout_length = self.cfg.rollout_length if not eval else self.cfg.eval_rollout_length
        result = self.rollout_func(
            eval,
            self,
            rollout_desc,
            env,
            behaving_policies,
            self.data_server,
            rollout_length=rollout_length,
            sample_length=self.cfg.sample_length,
            padding_length=self.cfg.padding_length,
            rollout_epoch=rollout_epoch
            # decaying_exploration_cfg=self.cfg.decaying_exploration
        )
        global_timer.time("policy_update_end", "rollout_end", "rollout")

        result["timer"] = copy.deepcopy(global_timer.elapses)
        global_timer.clear()

        return result

    def random_permute(self, rollout_desc: RolloutDesc):
        main_agent_id = rollout_desc.agent_id
        policy_distributions = rollout_desc.policy_distributions
        agent_ids = list(policy_distributions.keys())
        new_agent_ids = np.random.permutation(agent_ids)
        new_policy_distributions = {agent_id: policy_distributions[new_agent_ids[idx]] for idx, agent_id in
                                    enumerate(agent_ids)}
        new_main_idx = np.where(new_agent_ids == main_agent_id)[0][0]
        new_main_agent_id = agent_ids[new_main_idx]
        rollout_desc.agent_id = new_main_agent_id
        rollout_desc.policy_distributions = new_policy_distributions
        return rollout_desc

    def get_policies(self, policy_ids):
        policies = OrderedDict()
        for agent_id, policy_id in policy_ids.items():
            policy = self.agents[agent_id].policy_data[policy_id].policy
            policies[agent_id] = (policy_id, policy)
        return policies

    def get_mix_policies(self, policy_distributions):  # TODO(yan) can use AgentID(str)? Dict[str: Dict[str: float]]
        nonzero_prob_policy_id = OrderedDict()
        mix_policy_prob = None
        for agent_id, distribution in policy_distributions.items():
            if len(distribution) == 1:
                nonzero_prob_policy_id[agent_id] = list(distribution.keys())[0]
            else:
                mix_policy_prob = OrderedDict()
                for pid, prob in distribution.items():
                    if prob > 0:
                        mix_policy_prob[pid] = prob
                nonzero_prob_policy_id[agent_id] = mix_policy_prob

        if mix_policy_prob is not None:
            mix_policy = Q_mixing_wrapper(mix_policy_prob)

        policy_profile = OrderedDict()
        self.pull_mix_policies(nonzero_prob_policy_id)
        for agent_id, distribution in nonzero_prob_policy_id.items():
            # print(f'agent id = {agent_id}, distribution = {distribution}')
            if isinstance(distribution, str):  # when only one policy
                policy = self.agents[agent_id].policy_data[distribution].policy
                policy_profile[agent_id] = (distribution, policy)
            else:
                mix_pid = 'Mix_of_'
                for pid, prob in distribution.items():
                    policy = self.agents[agent_id].policy_data[pid].policy
                    mix_policy.register_policy({pid: policy})
                    mix_pid += f'({pid})_'
                policy_profile[agent_id] = (mix_pid, mix_policy)
        return policy_profile

    def pull_policies(self, policy_ids):
        for agent_id, policy_id in policy_ids.items():
            if policy_id not in self.agents[agent_id].policy_data:
                policy_desc = ray.get(self.policy_server.pull.remote(self.id, agent_id, policy_id, old_version=None))
                if policy_desc is None:
                    raise Exception("{} {} not found in policy server".format(agent_id, policy_id))
                self.agents[agent_id].policy_data[policy_id] = policy_desc
            else:
                old_policy_desc: PolicyDesc = self.agents[agent_id].policy_data[policy_id]
                policy_desc = ray.get(
                    self.policy_server.pull.remote(self.id, agent_id, policy_id, old_version=old_policy_desc.version))
                if policy_desc is not None:
                    self.agents[agent_id].policy_data[policy_id] = policy_desc

    def pull_mix_policies(self, policy_ids):
        for agent_id, policy_id in policy_ids.items():
            if isinstance(policy_id, OrderedDict) and len(policy_id) > 1:
                for pid, _ in policy_id.items():
                    if pid not in self.agents[agent_id].policy_data:
                        policy_desc = ray.get(self.policy_server.pull.remote(self.id, agent_id, pid, old_version=None))
                        if policy_desc is None:
                            raise Exception("{} {} not found in policy server".format(agent_id, pid))
                        self.agents[agent_id].policy_server[pid] = policy_desc
                    else:
                        old_policy_desc: PolicyDesc = self.agents[agent_id].policy_data[pid]
                        policy_desc = ray.get(
                            self.policy_server.pull.remote(self.id, agent_id, pid, old_version=old_policy_desc.version))
                        if policy_desc is not None:
                            self.agents[agent_id].policy_data[pid] = policy_desc
            else:
                if isinstance(policy_id, OrderedDict):  # TODO(yan): format mismatch
                    policy_id = list(policy_id.keys())[0]

                if policy_id not in self.agents[agent_id].policy_data:
                    policy_desc = ray.get(
                        self.policy_server.pull.remote(self.id, agent_id, policy_id, old_version=None))
                    if policy_desc is None:
                        raise Exception("{} {} not found in policy server".format(agent_id, policy_id))
                    self.agents[agent_id].policy_data[policy_id] = policy_desc
                else:
                    old_policy_desc: PolicyDesc = self.agents[agent_id].policy_data[policy_id]
                    policy_desc = ray.get(self.policy_server.pull.remote(self.id, agent_id, policy_id,
                                                                         old_version=old_policy_desc.version))
                    if policy_desc is not None:
                        self.agents[agent_id].policy_data[policy_id] = policy_desc

    def sample_policies(self, policy_distributions):
        policy_ids = OrderedDict()
        for agent_id, distribution in policy_distributions.items():
            policy_ids[agent_id] = self.sample_policy(distribution)
        return policy_ids

    def sample_policy(self, policy_distribution):
        policy_ids = list(policy_distribution.keys())
        policy_probs = np.array([policy_distribution[policy_id] for policy_id in policy_ids], dtype=np.float32)

        policy_probs = policy_probs / np.sum(policy_probs)
        policy_id = np.random.choice(a=policy_ids, p=policy_probs)
        return policy_id

    def sample_mix_policies(self, policy_distributions):
        policy_id = OrderedDict()
        for agent_id, distribution in policy_distributions.items():
            if len(distribution) == 1:
                policy_id[agent_id] = list(distribution.keys())[0]
            else:
                mix_policy = OrderedDict()
                for pid, prob in distribution.items():
                    if prob > 0:
                        mix_policy[pid] = prob
                policy_id[agent_id] = mix_policy
        return policy_id




