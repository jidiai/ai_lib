from typing import OrderedDict
import numpy as np
from utils.episode import EpisodeKey
from envs.base_aec_env import BaseAECEnv
from utils.desc.task_desc import RolloutDesc
from utils.naming import default_table_name
from utils.timer import global_timer


def rename_field(data, field, new_field):
    for agent_id, agent_data in data.items():
        if field in agent_data:
            field_data = agent_data.pop(field)
            agent_data[new_field] = field_data
    return data


def select_fields(data, fields):
    rets = {
        agent_id: {
            field: agent_data[field]
            for field in fields if field in agent_data
        }
        for agent_id, agent_data in data.items()
    }
    return rets


def clear(data, agent_ids):
    for agent_id in agent_ids:
        data[agent_id] = {}
    return data


def union(main, sub):
    def update_dict(dict1, dict2):
        d = {}
        d.update(dict1)
        d.update(dict2)
        return d

    rets = {
        agent_id: update_dict(main[agent_id], sub.get(agent_id, {}))
        for agent_id in main
    }
    return rets


def stack_step_data(step_data_list, bootstrap_data, padding_length=None):
    episode_data = {}
    length_to_pad = padding_length - len(step_data_list)
    active_masks = np.zeros((padding_length, 1, 1), dtype=float)
    assert EpisodeKey.ACTIVE_MASK not in step_data_list[0]
    active_masks[:len(step_data_list)] = 1
    episode_data[EpisodeKey.ACTIVE_MASK] = active_masks
    for field in step_data_list[0]:
        data_list = [step_data[field] for step_data in step_data_list]
        if field in bootstrap_data:
            data_list.append(bootstrap_data[field])
        data = np.stack(data_list)
        if padding_length is not None:
            pad_width = ((0, length_to_pad),) + ((0, 0),) * (len(data.shape) - 1)
            data = np.pad(data, pad_width, mode="edge")
        episode_data[field] = data
    return episode_data


def rollout_func(
        eval: bool,
        rollout_worker,
        rollout_desc: RolloutDesc,
        env: BaseAECEnv,
        behavior_policies,
        data_server,
        **kwargs
):
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """
    if not eval:
        assert data_server is not None
        # padding_length=kwargs.get("padding_length",None)
        # assert padding_length is not None
    render = kwargs.get("render", False)
    rollout_epoch = kwargs.get("rollout_epoch", 0)

    main_agent_id = rollout_desc.agent_id
    assert main_agent_id in behavior_policies

    policy_ids = OrderedDict()
    policies = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policies[agent_id] = policy

    custom_reset_config = {
        "feature_encoders": feature_encoders
    }

    # {agent_id:{field:value}}
    step_data = {
        agent_id: {}
        for agent_id in behavior_policies
    }

    global_timer.record("env_step_start")
    env.reset(custom_reset_config)
    global_timer.time("env_step_start", "env_step_end", "env_step")
    if render:
        env.render()

    rnn_states = {
        agent_id: policies[agent_id].get_initial_state(batch_size=env.num_players[agent_id])
        for agent_id in env.agent_ids
    }

    # fields
    # EpisodeKey.CUR_OBS # EpisodeKey.NEXT_OBS
    # EpisodeKey.ACTION_MASK
    # EpisodeKey.REWARD
    # EpisodeKey.DONE
    # EpisodeKey.ACTOR_RNN_STATE
    # EpisodeKey.CRITIC_RNN_STATE
    # EpisodeKey.ACTION
    # EpisodeKey.ACTION_DIST
    # EpisodeKey.STATE_VALUE

    step_data_list = []
    for step_ctr, player_id in enumerate(env.agent_id_iter()):
        agent_id = env.id_mapping(player_id)
        agent_data = env.get_curr_agent_data(agent_id)

        if not bool(agent_data[agent_id][EpisodeKey.DONE]):
            policy_inputs = union(agent_data, rnn_states)
            _, policy = behavior_policies[agent_id]
            global_timer.record("inference_start")
            policy_outputs = {}
            assert rollout_epoch is not None
            policy_outputs[agent_id] = policy.compute_action(**policy_inputs[agent_id],
                                                             explore=(not eval and agent_id == main_agent_id),
                                                             step=rollout_epoch)
            global_timer.time("inference_start", "inference_end", "inference")
            actions = {agent_id: policy_outputs[agent_id][EpisodeKey.ACTION]}
        else:
            policy_inputs = union(agent_data, rnn_states)
            policy_outputs = {agent_id: {}}
            actions = {agent_id: None}

        global_timer.record("env_step_start")
        if render and not bool(agent_data[agent_id][EpisodeKey.DONE]):
            print(actions)
            print()
        env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")
        if render and not bool(agent_data[agent_id][EpisodeKey.DONE]):
            env.render()

        # print(player_id,agent_data)

        if not eval:
            if main_agent_id == agent_id:
                step_data = union(policy_inputs, select_fields(policy_outputs,
                                                               [EpisodeKey.ACTION, EpisodeKey.ACTION_DIST,
                                                                EpisodeKey.STATE_VALUE]))
                # NOTE(jh): we need the next reward and done to compute return.
                if len(step_data_list) > 0:
                    step_data_list[-1][EpisodeKey.REWARD] = step_data[main_agent_id][EpisodeKey.REWARD]
                    step_data_list[-1][EpisodeKey.DONE] = step_data[main_agent_id][EpisodeKey.DONE]
                # append new data
                step_data_list.append(step_data[main_agent_id])
                # NOTE(jh): we need the next reward and done to compute return. Force them to be None here.
                step_data_list[-1][EpisodeKey.REWARD] = None
                step_data_list[-1][EpisodeKey.DONE] = None

                # update rnn states
        rnn_states = union(rnn_states,
                           select_fields(policy_outputs, [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE]))

    if not eval:
        # bootstrap_data=select_fields(
        #     {main_agent_id:step_data_list[-1]},
        #     [EpisodeKey.CUR_OBS,EpisodeKey.DONE,EpisodeKey.CRITIC_RNN_STATE,EpisodeKey.CUR_STATE]
        # )
        # bootstrap_data=bootstrap_data[main_agent_id]
        # episode=stack_step_data(step_data_list,bootstrap_data,padding_length=padding_length)
        # step_data_list.append(bootstrap_data)
        episode = step_data_list

        # if submit episodes.
        # data_server.save.remote(default_table_name(rollout_desc.agent_id,rollout_desc.policy_id,rollout_desc.share_policies),[episode])

        # elif submit transitions. NOTE(jh): frame stacking is not supported
        transitions = []
        for step in range(len(episode) - 1):
            # we need a newaxis to indicate length of step 1
            transition = {
                EpisodeKey.CUR_OBS: episode[step][EpisodeKey.CUR_OBS][np.newaxis, ...],
                EpisodeKey.ACTION_MASK: episode[step][EpisodeKey.ACTION_MASK][np.newaxis, ...],
                EpisodeKey.ACTION: episode[step][EpisodeKey.ACTION][np.newaxis, ...],
                EpisodeKey.REWARD: episode[step][EpisodeKey.REWARD][np.newaxis, ...],
                EpisodeKey.DONE: episode[step][EpisodeKey.DONE][np.newaxis, ...],
                EpisodeKey.NEXT_OBS: episode[step + 1][EpisodeKey.CUR_OBS][np.newaxis, ...],
                EpisodeKey.NEXT_ACTION_MASK: episode[step + 1][EpisodeKey.ACTION_MASK][np.newaxis, ...]
            }
            transitions.append(transition)
        data_server.save.remote(
            default_table_name(rollout_desc.agent_id, rollout_desc.policy_id, rollout_desc.share_policies), transitions)

    stats = env.get_episode_stats()

    return {"main_agent_id": main_agent_id, 'policy_ids': policy_ids, "stats": stats}