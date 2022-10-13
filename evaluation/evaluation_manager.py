from agent.policy_data.policy_data_manager import PolicyDataManager
from utils.desc.task_desc import RolloutEvalDesc
from utils.distributed import get_actor
import ray
import numpy as np

from utils.logger import Logger


class EvaluationManager:
    def __init__(self, cfg, agent_manager, policy_data_manager):
        self.cfg = cfg
        self.agents = agent_manager.agents
        self.policy_data_manager = policy_data_manager
        self.rollout_manager = get_actor("EvaluationManager", "RolloutManager")

    def eval(self, framework='psro'):
        # generate tasks from payoff matrix
        rollout_eval_desc = self.generate_rollout_tasks()

        # call rollout_eval remotely
        eval_results, extra_results = ray.get(self.rollout_manager.rollout_eval.remote(rollout_eval_desc))
        # Logger.info("eval_results: {}".format(eval_results))

        # update policy data
        if framework == 'psro':
            self.policy_data_manager.update_policy_data(eval_results, extra_results=extra_results)

    def _ordered(self, arr):
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return False
        return True

    def generate_rollout_tasks(self):
        payoff_matrix = self.policy_data_manager.get_matrix_data("payoff")
        indices = np.nonzero(payoff_matrix == self.policy_data_manager.cfg.fields.payoff.missing_value)

        policy_combs = []
        for index_comb in zip(*indices):
            if not self.agents.share_policies or self._ordered(index_comb):
                assert len(index_comb) == len(self.agents)
                policy_comb = {agent_id: {agent.policy_ids[index_comb[i]]: 1.0} for i, (agent_id, agent) in
                               enumerate(self.agents.items())}
                policy_combs.append(policy_comb)

        Logger.warning("Evaluation rollouts (num: {}) for {} policy combinations: {}".format(self.cfg.num_eval_rollouts,
                                                                                             len(policy_combs),
                                                                                             policy_combs))
        rollout_eval_desc = RolloutEvalDesc(policy_combs, self.cfg.num_eval_rollouts, self.agents.share_policies)
        return rollout_eval_desc