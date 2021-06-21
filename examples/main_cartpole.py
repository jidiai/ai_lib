from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import torch
import numpy as np
import random

if __name__ == '__main__':
    args = get_args()

    # 设置seed, 以便复现
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    env, args = make_env(args)

    runner = Runner(args, env)
    runner.run()