# -*- coding:utf-8  -*-
# Time  : 2021/7/27 下午3:58
# Author: Yahui Cui


import json


def magent_render(log):

    with open(log, 'r') as f:
        game_data = json.load(f)

    env_name = game_data["game_name"].split("-")[1]
    seed = game_data["seed"]
    map_size = game_data["map_size"]
    steps = game_data["steps"]

    import_path = "from pettingzoo.magent import " + env_name + " as env_core"
    exec(import_path, globals())
    func_name = "env_core"
    # env = battlefield_v3.parallel_env(map_size=50)
    env = eval(func_name).parallel_env(map_size=map_size)

    env.seed(seed)
    env.reset()

    for step in steps:
        action = step["joint_action"]
        _, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    """
    render step:
    1. run python run_log.py get log for MAgent
    2. run python render_from_log.py for MAgent local render
    """
    env_name = "magent-battle_v3-12v12"
    log_name = "logs/202107271558_magent-battle_v3-12v12.json"

    if env_name.split("-")[0] in ["magent"]:
        magent_render(log_name)
