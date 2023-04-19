from rollout.rollout_func_qmix import rollout_func

from utils.desc.task_desc import RolloutDesc, MARolloutDesc
from utils.episode import EpisodeKey
from envs.mpe.env import MPE
from buffer.data_server import DataServer
from utils.logger import Logger

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
        self.current_eps = 1

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
        assert box_list[0] == box_list[i], f"box list has unequal elements, {box_list[0] and box_list[i]}"
        low = np.concatenate([low, low])
        high = np.concatenate([high, high])
        total_shape += box_list[i].shape[0]

    return gym.spaces.Box(low=low,high=high, shape=(total_shape,), dtype =dtype)


INDEPENDENT_OBS = False

# env_cfg= #{'env_id': "simple_speaker_listener_v3"}
env_cfg={'env_id': "simple_reference_v2", "global_encoder": not INDEPENDENT_OBS}
# env_cfg = {'env_id': "simple_speaker_listener_v3", "global_encoder": not INDEPENDENT_OBS}
# env_cfg = {'env_id': "simple_spread_v2", "global_encoder": not INDEPENDENT_OBS}
# env_cfg={'env_id': "simple_v2"}

env = MPE(0,None,env_cfg)


if env_cfg['env_id']=='simple_reference_v2':
    agent_0_name = 'agent_0'
    agent_1_name = 'agent_1'
elif env_cfg['env_id']=='simple_speaker_listener_v3':
    agent_0_name = 'speaker_0'
    agent_1_name = 'listener_0'
elif env_cfg['env_id']=='simple_spread_v2':
    agent_0_name, agent_1_name, agent_2_name = env.agent_ids

if INDEPENDENT_OBS:
    ### Independent DQN player
    model_path = '/home/yansong/Downloads/test3/mpe_simple_reference/marl_dqn/Marl-2023-02-28-11-17-15/agent_0/agent_0_pop1_0/epoch_300000'
    from registry.registration import DeepQLearning
    # from utils.cfg import load_cfg
    # config_path='/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_simple_reference_dqn_marl.yaml'
    # dqn_cfg = load_cfg(config_path)
    policy = DeepQLearning.load(model_path)
    behavior_policies={
        aid: (f'policy_{i}', policy)
        for i, aid in enumerate([env.agent_ids[0]])
    }

else:

    from utils.cfg import load_cfg
    ### QMix
    config_path = '/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_simple_reference_qmix_marl.yaml'
    from registry.registration import QMix
    cfg = load_cfg(config_path)
    policy = QMix(registered_name='QMix',
                  observation_space=env.observation_spaces['agent_0'],
                  action_space=env.action_spaces['agent_0'],
                  model_config=cfg['populations'][0]['algorithm']['model_config'],
                  custom_config=cfg['populations'][0]['algorithm']['custom_config'])
    behavior_policies={
        aid: (f'policy_{i}', policy)
        for i, aid in enumerate([env.agent_ids[0]])
    }

rollout_desc = MARolloutDesc(["agent_0"], {'agent_0':['policy_1']}, None, True, None, None, None)


cfg.data_server.table_cfg.rate_limiter_cfg.min_size = 1
datasever = DataServer('dataserver_1', cfg.data_server)
table_name =  f'agent_0_{rollout_desc.policy_id["agent_0"][0]}'
datasever.create_table(table_name)

from registry.registration import QMixTrainer
trainer = QMixTrainer("qmix_trainer")



print(behavior_policies)
for _ in range(5):
    env = MPE(0, None, env_cfg)

    results=rollout_func(
        eval=False,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=datasever,
        rollout_length=25,
        render=False,
        rollout_epoch=0,
        episode_mode='traj'
    )

data_list = []
for _ in range(1):
    sample, _ = datasever.sample(table_name, batch_size = 2)
    data_list.append(sample)


def stack(samples):
    ret = {}
    for k, v in samples[0].items():
        # recursively stack
        if isinstance(v, dict):
            ret[k] = stack([sample[k] for sample in samples])
        elif isinstance(v, np.ndarray):
            ret[k] = np.stack([sample[k] for sample in samples])
        elif isinstance(v, list):
            ret[k] = [
                stack([sample[k][i] for sample in samples])
                for i in range(len(v))
            ]
        else:
            raise NotImplementedError
    return ret

#merge data
samples = []
for i in range(len(data_list[0])):
    sample = {}
    for data in data_list:
        sample.update(data[i])
    samples.append(sample)

stack_samples = stack(samples)

policy_0 = policy.to_device('cuda:0')

trainer.reset(policy_0, cfg.training_manager.trainer)
trainer.optimize(stack_samples)

Logger.info(f"Training complete")