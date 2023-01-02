from rollout.rollout_func_share import rollout_func
from utils.desc.task_desc import RolloutDesc
from utils.episode import EpisodeKey
from envs.mpe.env import MPE

import numpy as np

class DefaultFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces):

        self._action_space = action_spaces
        self._observation_space = observation_spaces

    def encode(self, state):
        # obs=np.array([self._policy.state_index(state)],dtype=int)
        # print(self._policy.state_index(state))
        obs = state
        action_mask = np.ones(self._action_space.n, dtype=np.float32)
        return obs, action_mask

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class RandomPlayer:
    def __init__(self, action_space, obs_space):
        self.action_space = action_space
        self.feature_encoder = DefaultFeatureEncoder(action_space,
                                                     obs_space)

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(1),
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(1),
        }

    def compute_action(self, **kwargs):
        obs = kwargs.get(EpisodeKey.CUR_OBS)
        action = []
        for _ in range(obs.shape[0]):
            action.append(self.action_space.sample())
        action = np.array(action)

        return {
            EpisodeKey.ACTION: action,
            EpisodeKey.CRITIC_RNN_STATE: kwargs[EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE],
        }


# env_cfg= #{'env_id': "simple_speaker_listener_v3"}
env_cfg={'env_id': "simple_reference_v2"}

env = MPE(0,None,env_cfg)

policy_set = {
    RandomPlayer(env.action_spaces(i),
                 env.observation_spaces(i))  for i in env.agent_ids
}

# agent_policy_mapping = {
#     ''
# }

rollout_desc = RolloutDesc("agent_0", None, None, True, None, None, None)
behavior_policies={
    aid: (f'policy_{i}', RandomPlayer(env.action_spaces(aid),
                 env.observation_spaces(aid)))
    for i, aid in enumerate([env.agent_ids[0]])
}

print(behavior_policies)

results=rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=None,
    rollout_length=25,
    render=True
)


print(results)






