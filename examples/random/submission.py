# -*- coding:utf-8  -*-


def single_action(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        for j in range(len(action_space_list_each)):
            each = [0] * action_space_list_each[j].n
            idx = action_space_list_each[j].sample()
            each[idx] = 1
            player.append(each)
    return player


def my_controller(observation_list, action_space_list, is_act_continuous):
    joint_action = []
    for i in range(len(observation_list)):
        player = single_action(action_space_list[0], is_act_continuous)
        joint_action.append(player)
    return joint_action
