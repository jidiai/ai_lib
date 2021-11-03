# -*- coding:utf-8  -*-
import os
import time
import json
import sys
import numpy as np

from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type
from pathlib import Path
CURRENT_PATH = str(Path(__file__).resolve().parent)
examples_path = os.path.join(CURRENT_PATH, "examples")
sys.path.append(examples_path)
print(sys.path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型：%s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i]

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]
            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            joint_action.append(each)
    print(joint_action)
    return joint_action


def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)


def render_game(g, fps=1):
    """
    This function is used to generate log for pygame rendering locally and render in time.
    The higher the fps, the faster the speed for rendering next step.
    only support gridgame:
    "gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_5p", "sokoban_1p", "cliffwalking"
    """

    import pygame
    pygame.init()
    screen = pygame.display.set_mode(g.grid.size)
    pygame.display.set_caption(g.game_name)
    clock = pygame.time.Clock()
    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/examples/algo/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = dict(game_name=env_type, n_player=g.n_player, board_height=g.board_height, board_width=g.board_width,
                     init_state=str(g.get_render_data(g.current_state)), init_info=str(g.init_info), start_time=st,
                     mode="window", render_info={"color": g.colors, "grid_unit": g.grid_unit, "fix": g.grid_unit_fix})

    all_observes = g.all_observes
    while not g.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        step = "step%d" % g.step_cnt
        print(step)
        game_info[step] = {}
        game_info[step]["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_space, all_observes)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        if info_before:
            game_info[step]["info_before"] = info_before
        game_info[step]["joint_action"] = str(joint_act)

        pygame.surfarray.blit_array(screen, g.render_board().transpose(1, 0, 2))
        pygame.display.flip()

        game_info[step]["state"] = str(g.get_render_data(g.current_state))
        game_info[step]["reward"] = str(reward)

        if info_after:
            game_info[step]["info_after"] = info_after

        clock.tick(fps)

    game_info["winner"] = g.check_win()
    game_info["winner_information"] = str(g.won)
    game_info["n_return"] = str(g.n_return)
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed


def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    log_path = os.getcwd() + '/logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = get_logger(log_path, g.game_name, json_file=render_mode)
    set_seed(g, env_name)

    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/examples/algo/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {"game_name": env_name,
                 "n_player": g.n_player,
                 "board_height": g.board_height if hasattr(g, "board_height") else None,
                 "board_width": g.board_width if hasattr(g, "board_width") else None,
                 "init_info": g.init_info,
                 "start_time": st,
                 "mode": "terminal",
                 "seed": g.seed if hasattr(g, "seed") else None,
                 "map_size": g.map_size if hasattr(g, "map_size") else None}

    steps = []
    all_observes = g.all_observes
    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        if g.step_cnt % 10 == 0:
            print(step)

        if hasattr(g, "env_core"):
            if hasattr(g.env_core, "render"):
                g.env_core.render()
        info_dict = {"time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
        joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)
        all_observes, reward, done, info_before, info_after = g.step(joint_act)
        if env_name.split("-")[0] in ["magent"]:
            info_dict["joint_action"] = g.decode(joint_act)
        if info_before:
            info_dict["info_before"] = info_before
        info_dict["reward"] = reward
        if info_after:
            info_dict["info_after"] = info_after
        steps.append(info_dict)

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)
    logger.info(logs)


def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'examples', 'algo')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":
    # "gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_2p", "snakes_5p", "sokoban_1p"
    # "classic_CartPole-v0", "classic_MountainCar-v0", "classic_MountainCarContinuous-v0",
    # "classic_Pendulum-v0", "classic_Acrobot-v1", "football_11v11_kaggle", "MiniWorld-Hallway-v0",
    # "MiniWorld-OneRoom-v0", "MiniWorld-OneRoomS6-v0", "MiniWorld-OneRoomS6Fast-v0",
    # "MiniWorld-TMaze-v0", "MiniWorld-TMazeLeft-v0", "MiniWorld-TMazeRight-v0", "MiniGrid-DoorKey-16x16-v0",
    # "MiniGrid-MultiRoom-N6-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0", "ParticleEnv-simple",
    # "ParticleEnv-simple_adversary", "ParticleEnv-simple_crypto", "ParticleEnv-simple_push",
    # "ParticleEnv-simple_reference", "ParticleEnv-simple_speaker_listener", "ParticleEnv-simple_spread",
    # "ParticleEnv-simple_tag", "ParticleEnv-simple_world_comm", "football_11_vs_11_stochastic",
    # "overcookedai-cramped_room", "overcookedai-asymmetric_advantages", "overcookedai-coordination_ring",
    # "overcookedai-forced_coordination", "overcookedai-counter_circuit", "magent-battle_v3-12v12",
    # "magent-battle_v3-20v20", "gridworld", "cliffwalking", "smarts-loop", "sc2-MoveToBeacon", "olympics_running",
    # "smarts-cloverleaf", "smarts-figure_eight", "smarts-minicity", "smarts-straight", "smarts-zoo_intersection",
    # "smarts-ngsim-partial", "smarts-ngsim", "robotics_FetchPickAndPlace", "robotics_HandManipulateBlock",
    # "robotics_HandManipulatePen", "sc2-CollectMineralShards", "sc2-FindAndDefeatZerglings", "sc2-DefeatRoaches",
    # "sc2-DefeatZerglingsAndBanelings", "sc2-Simple64", "chessandcard-go_v4", "chessandcard-chess_v3",
    # "chessandcard-checkers_v3", "chessandcard-mahjong_v3", "chessandcard-texas_holdem_v3",
    # "chessandcard-texas_holdem_no_limit_v3", "chinesechess"
    env_type = "olympics_running"
    game = make(env_type)

    # 针对"classic_"环境，使用gym core 进行render;
    # gridgame类环境（"gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3",
    # "snakes_5p", "sokoban_1p", "cliffwalking"），使用replay工具包的replay.html，通过上传.json进行网页回放
    render_mode = False

    # gridgame类环境支持实时render（"gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3",
    # "snakes_5p", "sokoban_1p", "cliffwalking"）
    render_in_time = False

    # print("可选policy 名称类型:", get_valid_agents())
    policy_list = ["random"] * len(game.agent_nums)

    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    if render_in_time:
        render_game(game)
    else:
        run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)
