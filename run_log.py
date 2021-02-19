# -*- coding:utf-8  -*-
import os
import time
import json
from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_joint_action_eval(game, player_ids, policy_list, actions_spaces):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型：%s" % str(obs_type))

        players_id_list = player_ids[policy_i]

        if game.obs_type[policy_i] == "grid":
            obs_list = game.get_grid_many_observation(game.current_state, players_id_list)
        elif game.obs_type[policy_i] == "vector":
            obs_list = game.get_vector_many_observation(game.current_state, players_id_list)
        elif game.obs_type[policy_i] == "dict":
            obs_list = game.get_dict_many_observation(game.current_state, players_id_list)

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        each = eval(function_name)(obs_list, action_space_list, game.is_act_continuous)
        if len(each) != game.agent_nums[policy_i]:
            error = "模型动作空间维度%d不正确！应该是%d" % (len(each), game.agent_nums[policy_i])
            raise Exception(error)

        joint_action.extend(each)
    return joint_action


def run_game(g, env_name, player_ids, actions_spaces, policy_list):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    log_path = os.getcwd() + '/logs/'
    logger = get_logger(log_path, g.game_name, json_file=render_mode)

    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/examples/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    if not g.is_obs_continuous:
        st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        game_info = {"game_name": env_name, "n_player": g.n_player, "board_height": g.board_height,
                     "board_width": g.board_width, "init_info": g.init_info,
                     "start_time": st,
                     "mode": "terminal"}

    steps = []
    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        if g.step_cnt % 10 == 0:
            print(step)
        if g.is_obs_continuous:
            if hasattr(g, "env_core") and render_mode:
                g.env_core.render()
        else:
            info_dict = {}
            info_dict["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        joint_act = get_joint_action_eval(g, player_ids, policy_list, actions_spaces)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        if not g.is_obs_continuous:
            if info_before:
                info_dict["info_before"] = info_before
            info_dict["reward"] = reward
            if info_after:
                info_dict["info_after"] = info_after
            steps.append(info_dict)

    if not g.is_obs_continuous:
        game_info["steps"] = steps
        game_info["winner"] = g.check_win()
        game_info["winner_information"] = g.won
        game_info["n_return"] = g.n_return
        ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        game_info["end_time"] = ed
        logs = json.dumps(game_info, ensure_ascii=False)
        logger.info(logs)


def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'examples')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":
    # "gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_5p", "sokoban_1p"
    # "classic_CartPole-v0", "classic_CartPole-v1", "classic_MountainCar-v0", "classic_MountainCarContinuous-v0",
    # "classic_Pendulum-v0", "classic_Acrobot-v1"
    env_type = "snakes_3v3"
    game = make(env_type, conf=None)

    # 针对"classic_"环境，使用gym core 进行render;
    # 非"classic_"环境，使用replay工具包的replay.html，通过上传.json进行网页回放
    render_mode = False

    print("可选policy 名称类型:", get_valid_agents())
    policy_list = ["random"] * len(game.agent_nums)

    player_id, actions_space = get_players_and_action_space_list(game)
    run_game(game, env_type, player_id, actions_space, policy_list)
