from algorithm.q_learning.policy import QLearning
from rollout.rollout_func_aec import rollout_func
from utils.desc.task_desc import RolloutDesc
from utils.episode import EpisodeKey
from envs.leduc_poker.env import LeducPokerEnv, DefaultFeatureEncoder
import numpy as np


class FeatureEncoder:
    def __init__(self):
        pass

    def encode(self, state):
        legal_action_idices = state.legal_actions()
        action_mask = np.zeros(3, dtype=np.float32)
        action_mask[legal_action_idices] = 1
        return state, action_mask


class HumanPlayer:
    def __init__(self):
        self.feature_encoder = FeatureEncoder()

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(1),
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(1),
        }

    def compute_action(self, **kwargs):
        obs = kwargs[EpisodeKey.CUR_OBS]
        action_mask = kwargs[EpisodeKey.ACTION_MASK][0]
        valid_actions = np.nonzero(action_mask)[0]
        action = input(
            "player {}: valid actions are {}, please input your action(0-fold,1-call,2-raise):".format(
                obs.current_player(), valid_actions
            )
        )
        action = int(action)

        return {
            EpisodeKey.ACTION: action,
            EpisodeKey.CRITIC_RNN_STATE: kwargs[EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE],
        }


model_path_0 = "expr_log/log/leduc_poker/test_q_learning/2022-09-24-21-19-46/agent_0/agent_0_default_1/best"
model_path_1 = "expr_log/log/leduc_poker/test_q_learning/2022-09-26-09-42-41/agent_0/agent_0_default_2/best"

policy_id_0 = "policy_0"
policy_id_1 = "policy_1"
policy_0 = QLearning.load(model_path_0, env_agent_id="agent_0")
policy_1 = QLearning.load(model_path_1, env_agent_id="agent_1")

env = LeducPokerEnv(0, None, None)
rollout_desc = RolloutDesc("agent_0", None, None, None)
behavior_policies = {
    "agent_0": (policy_id_0, policy_0),
    "agent_1": (policy_id_1, policy_1),
}

results = rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=None,
    padding_length=42,
    render=True,
)

print(results)
