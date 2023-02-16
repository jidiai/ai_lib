# from rollout.rollout_func_share import rollout_func
from rollout.rollout_func_share import rollout_func

from utils.desc.task_desc import RolloutDesc
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
        assert box_list[0] == box_list[i], f"box list has unequal elements, {box_list[0] and box_list[i]}"
        low = np.concatenate([low, low])
        high = np.concatenate([high, high])
        total_shape += box_list[i].shape[0]

    return gym.spaces.Box(low=low,high=high, shape=(total_shape,), dtype =dtype)


INDEPENDENT_OBS = True

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
    # model_path = '/home/yansong/Desktop/jidiai/ai_lib_V2_logs/simple_reference/dqn/new_trial/agent_0/agent_0_default_0/epoch_300000'
    # from registry.registration import DeepQLearning
    # from utils.cfg import load_cfg
    # config_path='/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_dqn_marl.yaml'
    # dqn_cfg = load_cfg(config_path)
    # policy = DeepQLearning.load(model_path)
    # behavior_policies={
    #     aid: (f'policy_{i}', policy)
    #     for i, aid in enumerate([env.agent_ids[0]])
    # }

    #### Independent DDPG player
    # from registry.registration import DDPG
    # from utils.cfg import load_cfg
    # config_path = '/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_ddpg_marl.yaml'
    # ddpg_cfg = load_cfg(config_path)
    # policy = DDPG(registered_name='DDPG',
    #               observation_space=env.observation_spaces('agent_0'),
    #               action_space=env.action_spaces('agent_0'),
    #               model_config=ddpg_cfg['populations'][0]['algorithm']['model_config'])

    ### Independent PPO player
    from registry.registration import PPO
    from utils.cfg import load_cfg
    config = '/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_simple_reference_ppo_marl.yaml'
    ppo_cfg = load_cfg(config)
    policy = PPO(registered_name='PPO',
                 observation_space=env.observation_spaces(agent_0_name),
                 action_space=env.action_spaces(agent_1_name),
                 model_config=ppo_cfg['populations'][0]['algorithm']['model_config'])


    behavior_policies={
        aid: (f'policy_{i}', policy)
        for i, aid in enumerate([env.agent_ids[0]])
    }

    #### Random player
    # policy_set = {
    #     RandomPlayer(env.action_spaces(i),
    #                  env.observation_spaces(i))  for i in env.agent_ids
    # }
    # behavior_policies={
    #     aid: (f'policy_{i}', RandomPlayer(env.action_spaces(aid),
    #                  env.observation_spaces(aid)))
    #     for i, aid in enumerate([env.agent_ids[0]])
    # }
else:
    # state_space = merge_gym_box([env.observation_spaces(aid)
    #                              for aid in env.agent_ids])
    # action_space = env.action_spaces(env.agent_ids[0])
    # from registry.registration import DeepQLearning
    # from utils.cfg import load_cfg
    # config_path='/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_dqn_marl.yaml'
    # dqn_cfg = load_cfg(config_path)
    # policy = DeepQLearning(
    #     registered_name='DeepQLearning',
    #
    # )
    model_path = '/home/yansong/Desktop/jidiai/ai_lib_V2_logs/simple_reference/madqn/new_trial/agent_0/agent_0_default_0/epoch_300000'
    from registry.registration import DeepQLearning
    from utils.cfg import load_cfg
    config_path='/home/yansong/Desktop/jidiai/ai_lib/expr/mpe/mpe_madqn_marl.yaml'
    dqn_cfg = load_cfg(config_path)
    policy = DeepQLearning.load(model_path)
    behavior_policies={
        aid: (f'policy_{i}', policy)
        for i, aid in enumerate([env.agent_ids[0]])
    }





rollout_desc = RolloutDesc("agent_0", None, None, True, None, None, None)




print(behavior_policies)

results=rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=None,
    rollout_length=25,
    render=True,
    rollout_epoch=0,
    episode_mode='time-step'
)


print(results)






