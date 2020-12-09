# -*- coding:utf-8  -*-
import os
import time
from examples.randomagent import my_controller
import json
from env.chooseenv import make

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


def get_joint_action_from_observation(env, agent_num_list):
    joint_action = []

    for policy_i in range(len(agent_num_list)):
        if policy_i == 0:
            players_id_list = range(agent_num_list[policy_i])
        else:
            players_id_list = range(agent_num_list[policy_i-1], agent_num_list[policy_i])
        obs_list = env.get_grid_many_observation(game.current_state, players_id_list)
        obs_space_list = env.get_grid_many_obs_space(players_id_list)
        action_space_list = [env.get_single_action_space(player_id) for player_id in players_id_list]

        p_actions = my_controller(obs_list, action_space_list, obs_space_list)
        joint_action.extend(p_actions)

    return joint_action


def render_game(g, fps=1):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode(g.grid.size)
    pygame.display.set_caption(g.game_name)
    clock = pygame.time.Clock()

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = dict(game_name=g.game_name, n_player=g.n_player, board_height=g.board_height, board_width=g.board_width,
                     init_state=str(g.get_render_data(g.current_state)), init_info=str(g.init_info), start_time=st,
                     mode="window", render_info={"color": g.colors, "grid_unit": g.grid_unit, "fix": g.grid_unit_fix})

    while not g.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        step = "step%d" % g.step_cnt
        print(step)
        game_info[step] = {}
        game_info[step]["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        # 每个policy 控制的agent数量
        agent_num_list = list(eval(game.agent_nums))
        # 根据agent number 分配 player id
        agent_num_list = get_players_id(agent_num_list)

        joint_act = get_joint_action_from_observation(g, agent_num_list)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        if info_before:
            game_info[step]["info_before"] = str(info_before)
        game_info[step]["joint_action"] = str(joint_act)

        pygame.surfarray.blit_array(screen, g.render_board().transpose(1, 0, 2))
        pygame.display.flip()

        game_info[step]["state"] = str(g.get_render_data(g.current_state))
        game_info[step]["reward"] = str(reward)

        if info_after:
            game_info[step]["info_after"] = str(info_after)

        clock.tick(fps)

    game_info["winner"] = g.check_win()
    game_info["winner_information"] = str(g.won)
    game_info["n_return"] = str(g.n_return)
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    json_object = json.dumps(game_info, indent=4, ensure_ascii=False)
    # print(json_object)


def get_players_id(agent_num_list):
    if sum(agent_num_list) != game.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(agent_num_list), game.n_player))

    for i in range(1, len(agent_num_list)):
        agent_num_list[i] += agent_num_list[i-1]
    return agent_num_list


if __name__ == "__main__":
    env_list = ["gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_5p"]
    env_type = "gobang_1v1"
    game = make(env_type)
    # 当前支持randomagent中的策略进行self play
    # policy_list = ["randomagent"] or ["randomagent","randomagent"]
    render_game(game)
