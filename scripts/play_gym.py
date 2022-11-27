from rollout.rollout_func_seg import rollout_func
from utils.desc.task_desc import RolloutDesc
from utils.episode import EpisodeKey
from envs.gym.env import GymEnv, DefaultGymFeatureEncoder
import numpy as np


class FeatureEncoder:
    def __init__(self, action_space):
        self._action_space = action_space

    def encoder(self, state):
        return self._action_space.sample, np.ones(
            self._action_space.n, dtype=np.float32
        )


class RandomPlayer:
    def __init__(self, action_space):
        self.action_space = action_space
        self.feature_encoder = FeatureEncoder(action_space)

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(1),
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(1),
        }

    def compute_action(self, **kwargs):
        action = self.action_space.sample()
        return {
            EpisodeKey.ACTION: action,
            EpisodeKey.CRITIC_RNN_STATE: kwargs[EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE],
        }


env_cfg = {"env_id": "CartPole-v0"}
env = GymEnv(0, None, env_cfg)
policy_0 = RandomPlayer(env.action_space)
policy_id_0 = "policy_0"

rollout_desc = RolloutDesc("agent_0", None, None, None, None, None, None)
behavior_policies = {
    "agent_0": (policy_id_0, policy_0),
}

results = rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=None,
    rollout_length=200,
    render=True,
)

print(results)
