from ast import List
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrainingDesc:
    agent_id: str
    policy_id: str
    policy_distributions: Dict
    kwargs: Dict


@dataclass
class RolloutDesc:
    agent_id: str
    policy_id: str
    # {agent_id:{"policy_ids":np.ndarray,"policy_probs":np.ndarray}}
    policy_distributions: Dict
    kwargs: Dict


@dataclass
class RolloutEvalDesc:
    policy_combinations: List[Dict]
    num_eval_rollouts: int
    kwargs: Dict


@dataclass
class PrefetchingDesc:
    table_name: str
    batch_size: int
