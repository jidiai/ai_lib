from dataclasses import dataclass
from typing import Any

@dataclass
class PolicyDesc:
    agent_id:str
    policy_id:str
    policy:Any=None
    version:int=0