import importlib
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
import os
import yaml
from types import SimpleNamespace as SN


def make_env(args):
    base_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_dir))
    env_wrapper_file_name = str("EnvWrapper." + str(args.scenario.replace('-', '_')))
    env_wrapper_file_import = importlib.import_module(env_wrapper_file_name)
    env = getattr(env_wrapper_file_import, str(args.scenario.replace('-', '_')))()
    return env


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
    yaml.dump(args, file)
    file.close()


def save_new_paras(args, save_path, file_name):
    file = open(os.path.join(str(save_path), str(file_name) + '.yaml'), mode='w', encoding='utf-8')
    yaml.dump(args.as_dict(), file)
    file.close()


def load_config(log_path, file_name):
    file = open(os.path.join(str(log_path), str(file_name) + '.yaml'), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


def get_paras_from_dict(config_dict):
    dummy_dict = config_reformat(config_dict)
    args = SN(**dummy_dict)
    return args


def config_reformat(my_dict):
    dummy_dict = {}
    for k, v in my_dict.items():
        if type(v) is dict:
            for k2, v2 in v.items():
                dummy_dict[k2] = v2
        else:
            dummy_dict[k] = v
    return dummy_dict