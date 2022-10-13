from utils.logger import Logger
from .utils.pretty_print import pformat_table
import numpy as np


def update_func(policy_data_manager, eval_results):
    assert policy_data_manager.agents.share_policies, "jh: assert symmetry"
    for policy_comb, agents_results in eval_results.items():
        agent_id_0, policy_id_0 = policy_comb[0]
        agent_id_1, policy_id_1 = policy_comb[1]
        results_0 = agents_results[agent_id_0]
        results_1 = agents_results[agent_id_1]

        idx_0 = policy_data_manager.agents[agent_id_0].policy_id2idx[policy_id_0]
        idx_1 = policy_data_manager.agents[agent_id_1].policy_id2idx[policy_id_1]

        if policy_data_manager.data["payoff"][idx_0, idx_1] == policy_data_manager.cfg.fields.payoff.missing_value:
            for key in ["payoff", "score", "win", "lose", "my_goal", "goal_diff"]:
                policy_data_manager.data[key][idx_0, idx_1] = 0
                policy_data_manager.data[key][idx_1, idx_0] = 0

        for key in ["score", "win", "lose", "my_goal", "goal_diff"]:
            policy_data_manager.data[key][idx_0, idx_1] += results_0[key] / 2
            policy_data_manager.data[key][idx_1, idx_0] += results_1[key] / 2
            if key == "score":
                policy_data_manager.data["payoff"][idx_0, idx_1] += results_0[key] - 0.5
                policy_data_manager.data["payoff"][idx_1, idx_0] += results_1[key] - 0.5

    # print data
    Logger.info("policy_data: {}".format(
        policy_data_manager.format_matrices_data(["payoff", "score", "win", "lose", "my_goal", "goal_diff"])))

    # pretty-print
    # support last_k. last_k=0 means showing all
    last_k = 10
    policy_ids_dict = {agent_id: agent.policy_ids[-last_k:] for agent_id, agent in policy_data_manager.agents.items()}
    policy_ids_0 = [policy_id.split("_")[-1] for policy_id in policy_ids_dict["agent_0"]]
    policy_ids_1 = [policy_id.split("_")[-1] for policy_id in policy_ids_dict["agent_1"]]

    payoff_matrix = policy_data_manager.get_matrix_data("payoff") * 100
    payoff_matrix = payoff_matrix[-last_k:, -last_k:]
    table = pformat_table(payoff_matrix, headers=policy_ids_1, row_indices=policy_ids_0, floatfmt="+3.0f")
    Logger.info("payoff table:\n{}".format(table))

    # TODO(jh): support viewing the most recent policy's battles against others ordered by payoff ascendingly
    worst_k = 10
    policy_ids_dict = {agent_id: agent.policy_ids for agent_id, agent in policy_data_manager.agents.items()}

    worst_indices = np.argsort(payoff_matrix[-1, :])[:worst_k]
    Logger.info("{}'s top {} worst opponents are:\n{}".format(
        policy_ids_dict["agent_0"][-1],
        worst_k,
        pformat_table(
            payoff_matrix[-1:, worst_indices].T,
            headers=["policy_id", "payoff"],
            row_indices=[policy_ids_dict["agent_1"][idx] for idx in worst_indices],
            floatfmt="+6.2f"
        )
    )
    )