from runner import Runner
from common.arguments import get_args
from common.utils import *

import torch
import numpy as np
import random

if __name__ == '__main__':
    args = get_args()
    print("================== args: ", args)
    print("== args.reload_config: ", args.reload_config, type(args.reload_config))

    # 保存训练参数 以便复现
    file_name = args.algo + "_" + args.scenario
    config_dir = os.path.join(os.getcwd(), "config")
    if args.reload_config:
        args = load_config(args, config_dir, file_name)

    # 设置训练seed
    torch.manual_seed(args.seed_nn)
    np.random.seed(args.seed_np)
    random.seed(args.seed_random)

    env, args = make_env(args)

    runner = Runner(args, env)
    runner.run()