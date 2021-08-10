# -*- coding:utf-8  -*-
import os
import time
import json

import numpy as np

from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type


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

        # if game.obs_type[policy_i] == "grid":
        #     obs_list = game.get_grid_many_observation(game.current_state, players_id_list, info_before)
        # elif game.obs_type[policy_i] == "vector":
        #     obs_list = game.get_vector_many_observation(game.current_state, players_id_list, info_before)
        # elif game.obs_type[policy_i] == "dict":
        #     obs_list = game.get_dict_many_observation(game.current_state, players_id_list, info_before)

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]
            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            # if len(each) != game.agent_nums[policy_i]:
            #     error = "模型%d动作空间维度%d不正确！应该是%d" % (int(t_agents_id[policy_i]), len(each), game.agent_nums[policy_i])
            #     raise Exception(error)
            joint_action.append(each)
    return joint_action


def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)


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

        import_path = '.'.join(file_path.split('/')[-4:])[:-3]
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
    info_before = ''
    all_observes = g.all_observes
    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        if g.step_cnt % 10 == 0:
            print(step)

        if hasattr(g, "env_core"):
            if hasattr(g.env_core, "render"):
                g.env_core.render()
        info_dict = {}
        info_dict["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
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
    # "gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_5p", "sokoban_1p"
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
    # "magent-battle_v3-20v20", "gridworld"
    env_type = "gridworld"
    game = make(env_type)

    # 针对"classic_"环境，使用gym core 进行render;
    # 非"classic_"环境，使用replay工具包的replay.html，通过上传.json进行网页回放
    render_mode = True

    # print("可选policy 名称类型:", get_valid_agents())
    policy_list = ["random"] * len(game.agent_nums)

    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)
