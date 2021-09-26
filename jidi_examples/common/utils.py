import importlib
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
import os
import yaml
from types import SimpleNamespace as SN
import matplotlib.pyplot as plt
import numpy as np


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


def plot_values(grid, values, colormap='pink', vmin=0, vmax=10):
    plt.imshow(values - 1000 * (grid < 0), interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])


def plot_action_values(algo, grid, action_values, vmin=-5, vmax=5):
    q = action_values
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    plot_values(grid, grid == 0, vmax=1)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 0:
                argmax_a = np.argmax(q[row, col])
                if argmax_a == 0:
                    x = col
                    y = row + 0.5
                    dx = 0
                    dy = -0.8
                if argmax_a == 1:
                    x = col - 0.5
                    y = row
                    dx = 0.8
                    dy = 0
                if argmax_a == 2:
                    x = col
                    y = row - 0.5
                    dx = 0
                    dy = 0.8
                if argmax_a == 3:
                    x = col + 0.5
                    y = row
                    dx = -0.8
                    dy = 0
                plt.arrow(x, y, dx, dy, width=0.02, head_width=0.4, head_length=0.4, length_includes_head=True, fc='k',
                          ec='k')
    plt.savefig("./assets/" + "grid_" + algo + ".png")
    plt.show()