from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingDesc:
    agent_id: str
    policy_id: str
    policy_distributions: Dict
    share_policies: bool
    sync: bool
    stopper: Any
    kwargs: Dict = field(default_factory=lambda: {})


@dataclass
class RolloutDesc:
    agent_id: str
    policy_id: str
    # {agent_id:{"policy_ids":np.ndarray,"policy_probs":np.ndarray}}
    policy_distributions: Dict
    share_policies: bool
    sync: bool
    stopper: Any
    type: str  # 'rollout', 'evaluation', 'simulation
    kwargs: Dict = field(default_factory=lambda: {})


@dataclass
class RolloutEvalDesc:
    policy_combinations: List[Dict]
    num_eval_rollouts: int
    share_policies: bool
    kwargs: Dict = field(default_factory=lambda: {})


@dataclass
class PrefetchingDesc:
    table_name: str
    batch_size: int
