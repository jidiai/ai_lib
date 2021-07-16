from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
import os
import yaml
from types import SimpleNamespace as SN


def make_env(args):
    env = make(args.scenario)
    if args.scenario == "classic_CartPole-v0":
        base_dir = Path(__file__).resolve().parent.parent.parent
        sys.path.append(str(base_dir))
        from EnvWrapper.classic_CartPole_v0 import Cartpole_v0
        env = Cartpole_v0()
    if args.scenario == "classic_MountainCar-v0":
        base_dir = Path(__file__).resolve().parent.parent.parent
        sys.path.append(str(base_dir))
        from EnvWrapper.classic_MountainCar_v0 import MountainCar_v0
        env = MountainCar_v0()
    action_space = env.get_actionspace()
    obs_space = env.get_observationspace()
    args.obs_space = obs_space
    args.action_space = action_space
    return env, args


def action_wrapper(action):
    joint_action_ = []
    action_a = action[0]
    each = [0] * 2
    each[action_a] = 1
    action_one_hot = [[each]]
    joint_action_.append([action_one_hot[0][0]])
    return joint_action_


def save_config(args, save_path, file_name):
    file = open(os.path.join(str(save_path), str(file_name) + '.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()


def load_config(args, log_path, file_name):
    file = open(os.path.join(str(log_path), str(file_name) + '.yaml'), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    args = SN(**config_dict)
    return args