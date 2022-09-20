from light_malib.rollout.rollout_func_aec import rollout_func
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.connect_four.env import ConnectFourEnv
from light_malib.algorithm.mappo.policy import MAPPO
import numpy as np

class FeatureEncoder:
    def __init__(self):
        pass
    
    def encode(self,observation,agent_id):
        return observation,np.ones(7)

class HumanPlayer:
    def __init__(self):
        self.feature_encoder=FeatureEncoder()
    
    def get_initial_state(self,batch_size):
        return {
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(1),
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(1)
        }
    
    def compute_action(self,**kwargs):
        action=input("input your action:")
        action=int(action)
        
        return {
            EpisodeKey.ACTION: action,
            EpisodeKey.CRITIC_RNN_STATE: kwargs[EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE]
        }

model_path_0="light_malib/trained_models/connect_four/test/PSRO_MAPPO_61/best"
model_path_1="light_malib/trained_models/connect_four/test/PSRO_MAPPO_61/best"

policy_id_0="policy_0"
policy_id_1="policy_1"
policy_0=MAPPO.load(model_path_0,env_agent_id="agent_0")
policy_1=HumanPlayer()#MAPPO.load(model_path_1,env_agent_id="agent_1")

env=ConnectFourEnv(0,None,None)
rollout_desc=RolloutDesc("agent_0",None,None,None)
behavior_policies={
    "agent_0": (policy_id_0,policy_0),
    "agent_1": (policy_id_1,policy_1)
}

rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=None,
    padding_length=42,
    render=True
)