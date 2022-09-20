from ensurepip import bootstrap
from typing import OrderedDict
import numpy as np
from light_malib.utils.logger import Logger
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.base_env import BaseEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_table_name

def rename_field(data,field,new_field):
    for agent_id,agent_data in data.items():
        field_data=agent_data.pop(field)
        agent_data[new_field]=field_data
    return data

def select_fields(data,fields):
    rets = { 
        agent_id: {
            field: agent_data[field]
            for field in fields if field in agent_data
        }
        for agent_id,agent_data in data.items()
    }
    return rets

def update_fields(data1,data2):
    def update_dict(dict1,dict2):
        d={}
        d.update(dict1)
        d.update(dict2)
        return d  
    rets = {
        agent_id: update_dict(data1[agent_id],data2[agent_id])
        for agent_id in data1
    }
    return rets

def stack_step_data(step_data_list,bootstrap_data):
    episode_data={}
    for field in step_data_list[0]:
        data_list=[step_data[field] for step_data in step_data_list]
        if field in bootstrap_data:
            data_list.append(bootstrap_data[field])
        episode_data[field]=np.stack(data_list)
    return episode_data

def rollout_func(
    eval: bool,
    rollout_worker,
    rollout_desc:RolloutDesc,
    env:BaseEnv,
    behavior_policies,
    data_server,
    rollout_length,
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

    # TODO(jh)
    # current_rollout_epoch = kwargs["rollout_epoch"]
    # decaying_exploration_cfg = kwargs['decaying_exploration_cfg']
    # init_noise = decaying_exploration_cfg['init_noise']
    # total_epoch_to_zero = decaying_exploration_cfg['total_epoch_to_zero']
    # interval = decaying_exploration_cfg['interval']       #length of interval at each exploation level

    # if policy.random_exploration:
    #     num_changes = total_epoch_to_zero / interval
    #     current_stage = (current_rollout_epoch + 1) // interval
    #     change_exp_each_time = init_noise/num_changes

    #     if change_exp_each_time*current_stage > init_noise:
    #         policy.random_exploration = 0
    #     else:
    #         policy.random_exploration = init_noise - change_exp_each_time*current_stage

    #     print(f'epoch = {current_rollout_epoch}, policy with radnom exploration {policy.random_exploration}')
    
    sample_length=kwargs.get("sample_length",rollout_length)
    render=kwargs.get("render",False)
    if render:
        env.render()

    policy_ids=OrderedDict()
    feature_encoders=OrderedDict()
    for agent_id, (policy_id,policy) in behavior_policies.items():
        feature_encoders[agent_id]=policy.feature_encoder
        policy_ids[agent_id]=policy_id
    
    custom_reset_config={
        "feature_encoders": feature_encoders,
        "main_agent_id": rollout_desc.agent_id,
        "rollout_length": rollout_length
    }
    # {agent_id:{field:value}}
    global_timer.record("env_step_start")
    env_rets = env.reset(custom_reset_config)
    global_timer.time("env_step_start","env_step_end","env_step")

    init_rnn_states={
        agent_id: behavior_policies[agent_id][1].get_initial_state(batch_size=env.num_players[agent_id]) 
        for agent_id in env.agent_ids
    }
    
    # TODO(jh): support multi-dimensional batched data based on dict & list using sth like NamedIndex.
    step_data=update_fields(env_rets,init_rnn_states)
    
    step=0
    step_data_list=[]
    while not env.is_terminated():      # XXX(yan): terminate only when step_length >= fragment_length
        # prepare policy input
        policy_inputs=rename_field(step_data,EpisodeKey.NEXT_OBS,EpisodeKey.CUR_OBS)
        policy_outputs={}
        global_timer.record("inference_start")
        for agent_id,(policy_id,policy) in behavior_policies.items():
            policy_outputs[agent_id] = policy.compute_action(**policy_inputs[agent_id])
        global_timer.time("inference_start","inference_end","inference")
        
        actions=select_fields(policy_outputs,[EpisodeKey.ACTION])
        
        global_timer.record("env_step_start")
        env_rets = env.step(actions)
        global_timer.time("env_step_start","env_step_end","env_step")
        
        # record data after env step
        step_data=update_fields(step_data,select_fields(env_rets,[EpisodeKey.REWARD,EpisodeKey.DONE]))
        step_data=update_fields(step_data,select_fields(policy_outputs,[EpisodeKey.ACTION,EpisodeKey.ACTION_DIST,EpisodeKey.STATE_VALUE]))
        
        # save data of trained agent for training
        step_data_list.append(step_data[rollout_desc.agent_id])

        # record data for next step
        step_data=update_fields(env_rets,select_fields(policy_outputs,[EpisodeKey.ACTOR_RNN_STATE,EpisodeKey.CRITIC_RNN_STATE]))
        
        step+=1
        if not eval:
            assert data_server is not None
            if step%sample_length==0:
                
                submit_ctr=step//sample_length
                submit_max_num=rollout_length//sample_length
                
                s_idx=sample_length*(submit_ctr-1)
                e_idx=sample_length*submit_ctr
                
                
                bootstrap_data=select_fields(step_data,[EpisodeKey.NEXT_OBS,EpisodeKey.DONE,EpisodeKey.CRITIC_RNN_STATE,EpisodeKey.CUR_STATE])
                bootstrap_data=bootstrap_data[rollout_desc.agent_id]
                bootstrap_data[EpisodeKey.CUR_OBS]=bootstrap_data[EpisodeKey.NEXT_OBS]
                
                episode = stack_step_data(
                    step_data_list[s_idx:e_idx],
                    # TODO CUR_STATE is not supported now
                    bootstrap_data
                )
        
                # submit data:
                data_server.save.remote(default_table_name(rollout_desc.agent_id,rollout_desc.policy_id,rollout_desc.share_policies),[episode])
                
                if submit_ctr!=submit_max_num:
                    # update model:
                    rollout_worker.pull_policies(policy_ids)
                    behavior_policies=rollout_worker.get_policies(policy_ids)
                    
    stats=env.get_episode_stats()
    
    return {"main_agent_id":rollout_desc.agent_id, 'policy_ids': policy_ids, "stats": stats}