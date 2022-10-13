from dataclasses import dataclass
from typing import Any, Union

@dataclass
class PolicyDesc:
    agent_id:str
    policy_id:str
    policy:Any
    version:Union[int,float]