import numpy as np
from env.chooseenv import make

def random_agent(observation_list, action_space_list, obs_space_list, model=None, is_act_continuous=False):
    joint_action = []
    for i in range(len(action_space_list)):
        player = sample(action_space_list[i], is_act_continuous)
        joint_action.append(player)
    return joint_action

def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            each = [0] * action_space_list_each[j].n
            idx = action_space_list_each[j].sample()
            each[idx] = 1
            player.append(each)
    return player

def my_controller(observation_list, action_space_list, obs_space_list):
    joint_action = random_agent(observation_list, action_space_list, obs_space_list,
                                model=None, is_act_continuous=False)
    return joint_action
