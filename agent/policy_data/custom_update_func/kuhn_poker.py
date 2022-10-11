from collections import OrderedDict
from light_malib.utils.distributed import get_actor
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger
from .utils.pretty_print import pformat_table
import numpy as np
import pyspiel
from open_spiel.python import algorithms
from open_spiel.python import policy as policy_lib
import ray
from open_spiel.python.algorithms.psro_v2.utils import aggregate_policies
from light_malib.framework.meta_solver.nash import Solver as NashSolver

def update_func(policy_data_manager,eval_results, **kwargs):
    assert policy_data_manager.agents.share_policies,"jh: assert symmetry"
    for policy_comb,agents_results in eval_results.items():
        agent_id_0, policy_id_0 = policy_comb[0]
        agent_id_1, policy_id_1 = policy_comb[1]
        results_0=agents_results[agent_id_0]
        results_1=agents_results[agent_id_1]
        
        idx_0=policy_data_manager.agents[agent_id_0].policy_id2idx[policy_id_0]
        idx_1=policy_data_manager.agents[agent_id_1].policy_id2idx[policy_id_1]

        if policy_data_manager.data["payoff"][idx_0,idx_1]==policy_data_manager.cfg.fields.payoff.missing_value:
            for key in ["payoff","score","win","lose","reward"]:
                policy_data_manager.data[key][idx_0,idx_1]=0
                policy_data_manager.data[key][idx_1,idx_0]=0
                
        for key in ["score","win","lose","reward"]:
            policy_data_manager.data[key][idx_0,idx_1]+=results_0[key]/2
            policy_data_manager.data[key][idx_1,idx_0]+=results_1[key]/2
            if key=="reward":
                policy_data_manager.data["payoff"][idx_0,idx_1]+=results_0[key]/2
                policy_data_manager.data["payoff"][idx_1,idx_0]+=results_1[key]/2
                if idx_0==idx_1:
                    # force to be 0 to avoid numeric issue?
                    assert (results_0[key]+results_1[key]-1.0)<1e-6
                    policy_data_manager.data["payoff"][idx_0,idx_1]=0
                
    # print data
    Logger.info("policy_data: {}".format(policy_data_manager.format_matrices_data(["payoff","score","win","lose","reward"])))
    
    # pretty-print
    # support last_k. last_k=0 means showing all
    last_k=10
    policy_ids_dict={agent_id:agent.policy_ids[-last_k:] for agent_id,agent in policy_data_manager.agents.items()}
    policy_ids_0=[policy_id.split("_")[-1] for policy_id in policy_ids_dict["agent_0"]]
    policy_ids_1=[policy_id.split("_")[-1] for policy_id in policy_ids_dict["agent_1"]]
    
    payoff_matrix=policy_data_manager.get_matrix_data("payoff")

    monitor=get_actor(policy_data_manager.id,"Monitor")
    training_agent_id = policy_data_manager.agents.training_agent_ids[0]
    pid = policy_data_manager.agents[training_agent_id].policy_ids[-last_k:]
    ray.get(monitor.add_array.remote("PSRO/Nash_Equilibrium/Payoff Table",payoff_matrix, pid, pid, payoff_matrix.shape[0], 'bwr', show_text=False))

    payoff_matrix=payoff_matrix[-last_k:,-last_k:]
    table=pformat_table(payoff_matrix,headers=policy_ids_1,row_indices=policy_ids_0,floatfmt="+4.2f")
    Logger.info("payoff table(reward):\n{}".format(table))


    # TODO(jh): support viewing the most recent policy's battles against others ordered by payoff ascendingly
    worst_k=10
    policy_ids_dict={agent_id:agent.policy_ids for agent_id,agent in policy_data_manager.agents.items()}
    
    worst_indices=np.argsort(payoff_matrix[-1,:])[:worst_k]
    Logger.info("{}'s top {} worst opponents are:\n{}".format(
            policy_ids_dict["agent_0"][-1],
            worst_k,
            pformat_table(
                payoff_matrix[-1:,worst_indices].T,
                headers=["policy_id","payoff"],
                row_indices=[policy_ids_dict["agent_1"][idx] for idx in worst_indices],
                floatfmt="+6.2f"
            )
        )
    )
    
    # compute exploitabilities
    training_agent_id=policy_data_manager.agents.training_agent_ids[0]
    exploitability_array=policy_data_manager.get_array_data("exploitability")[training_agent_id]
    missing_indices=np.nonzero(exploitability_array==policy_data_manager.cfg.fields.exploitability.missing_value)[0]
    missing_policy_ids=[policy_data_manager.agents[training_agent_id].policy_ids[missing_index] for missing_index in missing_indices]
    missing_policy_descs=ray.get([policy_data_manager.policy_server.pull.remote(policy_data_manager.id,training_agent_id,policy_id)
                                  for policy_id in missing_policy_ids])
    missing_policies=[policy_desc.policy for policy_desc in missing_policy_descs]
    for policy_id,policy in zip(missing_policy_ids,missing_policies):
        _policy=convert_to_official_tabular_policy(policy)
        exploitability=compute_exploitability(_policy)
        idx=policy_data_manager.agents[training_agent_id].policy_id2idx[policy_id]
        policy_data_manager.data["exploitability"][training_agent_id][idx]=exploitability

    exploitability_array=policy_data_manager.get_array_data("exploitability")[training_agent_id]   
    exploitability_array=exploitability_array[-last_k:].reshape(-1,1)
    Logger.info("last {} policies' exploitabilities are:\n{}".format(
            last_k,
            pformat_table(
                exploitability_array,
                headers=["policy_id","exploitability"],
                row_indices=policy_data_manager.agents[training_agent_id].policy_ids[-last_k:],
                floatfmt="+6.2f"
            )
        )
    )    
    
    # TODO(jh): we should compute exploitaibility of the nash eq?
    training_agent_id=policy_data_manager.agents.training_agent_ids[0]
    population_id="default"
   
    agent_id2policy_ids=OrderedDict()
    agent_id2policy_indices=OrderedDict()
    for agent_id in policy_data_manager.agents.keys():
        population=policy_data_manager.agents[agent_id].populations[population_id]
        agent_id2policy_ids[agent_id]=population.policy_ids
        agent_id2policy_indices[agent_id]=np.array([policy_data_manager.agents[agent_id].policy_id2idx[policy_id] for policy_id in population.policy_ids])
        
    # get payoff matrix
    payoff_matrix=policy_data_manager.get_matrix_data("payoff",agent_id2policy_indices)
        
    # compute nash
    equlibrium_distributions=NashSolver().compute(payoff_matrix)
    
    policy_distributions={}
    for probs,(agent_id,policy_ids) in zip(equlibrium_distributions,agent_id2policy_ids.items()):
        policy_distributions[agent_id]=OrderedDict(zip(policy_ids,probs))
    policy_descs=ray.get([policy_data_manager.policy_server.pull.remote(policy_data_manager.id,training_agent_id,policy_id)
                          for policy_id in policy_distributions[training_agent_id]])
    policies=[policy_desc.policy for policy_desc in policy_descs]
    probs=list(policy_distributions[training_agent_id].values())
    nash_eq_policy=get_nash_equilibrium_policy(policies,probs)
    nash_eq_exploitability=compute_exploitability(nash_eq_policy)
    if not hasattr(policy_data_manager,"nash_eq_exploitabilities"):
        policy_data_manager.nash_eq_exploitabilities=[]
    policy_data_manager.nash_eq_exploitabilities.append(nash_eq_exploitability)
    monitor=get_actor(policy_data_manager.id,"Monitor")
    ray.get(monitor.add_scalar.remote("PSRO/Nash_Equilibrium/Exploitability",nash_eq_exploitability,len(policy_data_manager.nash_eq_exploitabilities)-1))
    ray.get(monitor.add_scalar.remote("PSRO/Nash_Equilibrium/Exploitability_per_training_step", nash_eq_exploitability, kwargs['extra_results']['total_training_steps']))

    try:
        Logger.info("last {} nash eq' exploitabilities are:\n{}".format(
                last_k,
                pformat_table(
                    np.array(policy_data_manager.nash_eq_exploitabilities[-last_k:]).reshape(-1,1),
                    headers=["policy_id (nash eq)","exploitability"],
                    row_indices=policy_data_manager.agents[training_agent_id].policy_ids[-last_k:],
                    floatfmt="+6.2f"
                )
            )
        )
    except Exception as e:
        Logger.error(f"{e}")

def print_policy(policy):
    import itertools as it
    for state, probs in zip(it.chain(*policy.states_per_player),policy.action_probability_array):
        print(f'{state:6}   p={probs}')

def convert_to_official_tabular_policy(policy):
    game=pyspiel.load_game("kuhn_poker")
    official_tabular_policy=policy_lib.TabularPolicy(game)
    states=official_tabular_policy.states
    encoded=[policy.encoder.encode(state) for state in states]
    observations,action_masks=zip(*encoded)
    observations=np.array(observations)
    action_masks=np.array(action_masks)
    ret=policy.compute_action(**{EpisodeKey.CUR_OBS:observations,EpisodeKey.ACTION_MASK:action_masks},explore=False)
    action_probs=ret[EpisodeKey.ACTION_PROBS]
    assert official_tabular_policy.action_probability_array.shape==action_probs.shape
    official_tabular_policy.action_probability_array=action_probs
    assert np.all(action_probs.sum(axis=-1)==1),"{} {}".format(action_probs.shape,action_probs.sum(axis=-1))
    #print_policy(official_tabular_policy)
    return official_tabular_policy

def compute_exploitability(official_tabular_policy):
    game=pyspiel.load_game("kuhn_poker")
    exploitability=algorithms.exploitability.nash_conv(game,official_tabular_policy)
    return exploitability

def get_nash_equilibrium_policy(policies,probs):
    game=pyspiel.load_game("kuhn_poker")
    policies=[convert_to_official_tabular_policy(policy) for policy in policies]
    probs=np.array(probs)
    policy=aggregate_policies(game,total_policies=[policies,policies],probabilities_of_playing_policies=[probs,probs])
    return policy