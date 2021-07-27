from runner import Runner
# from common.arguments import get_args
import argparse
from common.utils import *

import torch
import numpy as np
import random

if __name__ == '__main__':
    # set env and algo
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="classic_MountainCar-v0", type=str)
    parser.add_argument('--algo', default="dqn", type=str, help="dqn/ppo/a2c/ddpg/ac/ddqn/duelingq/sac")

    parser.add_argument('--reload_config', action='store_true')  # 加是true；不加为false
    args = parser.parse_args()

    print("================== args: ", args)
    print("== args.reload_config: ", args.reload_config)

    runner = Runner(args)
    runner.run()