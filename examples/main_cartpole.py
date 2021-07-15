from runner import Runner
from common.arguments import get_args
from common.utils import make_env

import torch
import numpy as np
import random

if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(args.seed_nn)
    np.random.seed(args.seed_np)
    random.seed(args.seed_random)

    env, args = make_env(args)

    runner = Runner(args, env)
    runner.run()