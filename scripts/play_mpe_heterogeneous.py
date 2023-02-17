# from rollout.rollout_func_share import rollout_func
from rollout.rollout_func_independent import rollout_func

from utils.desc.task_desc import RolloutDesc, MARolloutDesc
from utils.episode import EpisodeKey
from envs.mpe.env import MPE

import numpy as np
import gym

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


def merge_gym_box(box_list):
    length = len(box_list)
    total_shape = box_list[0].shape[0]
    low = box_list[0].low
    high = box_list[0].high
    dtype = box_list[0].dtype

    for i in range(1,length):
        low = np.concatenate([low, box_list[i].low])
        high = np.concatenate([high, box_list[i].high])
        total_shape += box_list[i].shape[0]

    return gym.spaces.Box(low=low,high=high, shape=(total_shape,), dtype =dtype)


INDEPENDENT_OBS = False

# env_cfg= #{'env_id': "simple_speaker_listener_v3"}
# env_cfg={'env_id': "simple_reference_v2", "global_encoder": not INDEPENDENT_OBS}
env_cfg = {'env_id': "simple_speaker_listener_v3", "global_encoder": not INDEPENDENT_OBS}
# env_cfg = {'env_id': "simple_spread_v2", "global_encoder": not INDEPENDENT_OBS}
# env_cfg={'env_id': "simple_v2"}

env = MPE(0,None,env_cfg)

state_space = merge_gym_box([env.observation_spaces(aid)
                             for aid in env.possible_agents])

if env_cfg['env_id']=='simple_reference_v2':
    agent_0_name = 'agent_0'
    agent_1_name = 'agent_1'
elif env_cfg['env_id']=='simple_speaker_listener_v3':
    agent_0_name = 'speaker_0'
    agent_1_name = 'listener_0'
elif env_cfg['env_id']=='simple_spread_v2':
    agent_0_name, agent_1_name, agent_2_name = env.agent_ids


model_path = '/home/yansong/Desktop/jidiai/ai_lib_V2_logs/simple_reference/madqn/new_trial/agent_0/agent_0_default_0/epoch_300000'
from registry.registration import DeepQLearning
from utils.cfg import load_cfg
config_path='/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_simple_speaker_listener_madqn_marl.yaml'
dqn_cfg = load_cfg(config_path)

speaker_policy = DeepQLearning(registered_name='DeepQLearning',
                               observation_space=env.observation_spaces(agent_0_name),
                               action_space = env.action_spaces(agent_0_name),
                               model_config=dqn_cfg['populations'][0]['algorithm']['model_config'],
                               custom_config=dqn_cfg['populations'][0]['algorithm']['custom_config'])

listener_policy = DeepQLearning(registered_name='DeepQLearning',
                               observation_space=env.observation_spaces(agent_1_name),
                               action_space = env.action_spaces(agent_1_name),
                               model_config=dqn_cfg['populations'][1]['algorithm']['model_config'],
                               custom_config=dqn_cfg['populations'][1]['algorithm']['custom_config'])


behavior_policies = {agent_0_name: ('speaker_0_v1', speaker_policy),
                     agent_1_name: ('listener_0_v1', listener_policy)}

# behavior_policies={
#     aid: (f'policy_{i}', policy)
#     for i, aid in enumerate([env.agent_ids])
# }


agent_id_list = ['speaker_0', 'listener_0']
policy_id_list = [['speaker_0_v1'], ['listener_0_v1']]

# rollout_desc = RolloutDesc("agent_0", None, None, True, None, None, None)
rollout_desc = MARolloutDesc(agent_id=agent_id_list,
                             policy_id=dict(zip(agent_id_list, policy_id_list)),
                             policy_distributions={'speaker_0': {'speaker_0_v1': 1.},
                                                   "listener_0": {'listener_0_v1': 1.}},
                             share_policies=False,sync=True,stopper=None, type='evaluation')



from buffer import DataServer
from utils.naming import default_table_name

data_server = DataServer('data_server', dqn_cfg['data_server'])

for aid in agent_id_list:
    data_server.create_table(f'{aid}')



results=rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=data_server,
    rollout_length=25,
    render=True,
    rollout_epoch=0,
    episode_mode='traj'
)


print(results)






