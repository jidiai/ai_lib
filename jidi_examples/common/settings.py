# -*- coding:utf-8  -*-
# Time  : 2021/07/26 12:00
# Author: Yutong Wu

import attr, cattr
import copy
import os
import yaml
from types import SimpleNamespace as SN

from typing import (
    Dict,
    Any
)


@attr.s(auto_attribs=True)
class HyperparamSettings:
    use_network: bool = True
    marl: bool = False


@attr.s(auto_attribs=True)
class ExportableSettings:
    def as_dict(self):
        return cattr.unstructure(self)


@attr.s(auto_attribs=True)
class TABULARQSettings(HyperparamSettings):
    lr: float = 0.1
    buffer_capacity: int = 1
    gamma: float = 0.9
    epsilon: float = 0.2
    epsilon_end: float = 0.01
    learn_freq: int = 1
    learn_terminal: bool = False
    use_network: bool = False


@attr.s(auto_attribs=True)
class SARSASettings(HyperparamSettings):
    lr: float = 0.1
    buffer_capacity: int = 1
    gamma: float = 0.9
    epsilon: float = 0.2
    epsilon_end: float = 0.01
    learn_freq: int = 1
    learn_terminal: bool = False
    use_network: bool = False


@attr.s(auto_attribs=True)
class DQNSettings(HyperparamSettings):
    c_lr: float = 0.005
    buffer_capacity: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    epsilon: float = 0.5
    epsilon_end: float = 0.05
    target_replace: int = 100
    hidden_size: int = 64
    learn_freq: int = 1
    learn_terminal: bool = False


@attr.s(auto_attribs=True)
class DDQNSettings(HyperparamSettings):
    c_lr: float = 0.005
    buffer_capacity: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    epsilon: float = 0.5
    epsilon_end: float = 0.05
    target_replace: int = 100
    hidden_size: int = 64
    learn_freq: int = 1
    learn_terminal: bool = False


@attr.s(auto_attribs=True)
class DUELINGQSettings(HyperparamSettings):
    c_lr: float = 0.005
    buffer_capacity: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    epsilon: float = 0.5
    epsilon_end: float = 0.05
    target_replace: int = 100
    hidden_size: int = 64
    learn_freq: int = 1
    learn_terminal: bool = False


@attr.s(auto_attribs=True)
class DDPGSettings(HyperparamSettings):
    a_lr: float = 0.005
    c_lr: float = 0.005
    buffer_capacity: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.2
    epsilon: float = 0.5
    epsilon_end: float = 0.05
    hidden_size: int = 64
    update_freq: int = 5
    learn_freq: int = 10
    learn_terminal: bool = False


@attr.s(auto_attribs=True)
class TD3Settings(HyperparamSettings):
    a_lr: float = 0.0003
    batch_size: int = 100
    buffer_capacity: int = 10240
    c_lr: float = 0.0003
    epsilon: float = 0.3
    epsilon_end: float = 0.05
    exploration_noise: float = 0.1
    gamma: float = 0.99
    hidden_size: int = 100
    noise_clip: float = 0.5
    policy_delay: int = 2
    policy_noise: float = 0.2
    target_replace: int = 100
    tau: float = 0.005
    update_freq: int = 10
    marl: bool = False
    learn_freq: int = 10
    learn_terminal: bool = False


@attr.s(auto_attribs=True)
class PGSettings(HyperparamSettings):
    lr: float = 0.005
    buffer_capacity: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    hidden_size: int = 64
    learn_terminal: bool = True


@attr.s(auto_attribs=True)
class ACSettings(HyperparamSettings):
    c_lr: float = 0.005
    buffer_capacity: int = 100
    batch_size: int = 32
    gamma: float = 0.99
    hidden_size: int = 64
    learn_terminal: bool = True


@attr.s(auto_attribs=True)
class PPOSettings(HyperparamSettings):
    c_lr: float = 0.005
    a_lr: float = 0.005
    clip_param: float = 0.2
    max_grad_norm: float = 0.5
    update_freq: int = 10
    buffer_capacity: int = 100
    batch_size: int = 32
    gamma: float = 0.99
    hidden_size: int = 64
    learn_terminal: bool = True


@attr.s(auto_attribs=True)
class PPO_CNNSettings(PPOSettings):
    cnn_encoder: bool = True


@attr.s(auto_attribs=True)
class SACSettings(HyperparamSettings):
    c_lr: float = 0.0001
    a_lr: float = 0.0001
    alpha_lr: float = 0.0001
    target_entropy_ratio: float = 0.7
    buffer_capacity: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    target_replace: int = 100
    tune_entropy: bool = False
    alpha: float = 0.2
    epsilon: float = 0.2
    epsilon_end: float = 0.05
    hidden_size: int = 128
    num_hid_layer: int = 1
    learn_freq: int = 1
    learn_terminal: bool = False
    tau: float = 0.005
    policy_type: str = "discrete"


@attr.s(auto_attribs=True)
class MADDPGSettings(HyperparamSettings):
    lr_c: float = 0.01
    lr_a: float = 0.01
    buffer_capacity: int = 1000000
    batch_size: int = 1256
    gamma: float = 0.97
    tao: float = 0.01
    target_replace: int = 100
    marl: bool = True
    action_continuous: bool = True


@attr.s(auto_attribs=True)
class EnvSettingDefault:
    scenario: str = "classic_CartPole-v0"
    obs_space: int = 100
    action_space: int = 100
    obs_continuous: bool = True
    action_continuous: bool = False
    n_player: int = 1


@attr.s(auto_attribs=True)
class TrainingDefault:
    max_episodes: int = 1000
    evaluate_rate: int = 50
    render: bool = False
    save_interval: int = 100


@attr.s(auto_attribs=True)
class SeedSetting:
    seed_nn: int = 1
    seed_np: int = 1
    seed_random: int = 1


@attr.s(auto_attribs=True)
class TrainerSettings(ExportableSettings):
    hyperparameters: HyperparamSettings = attr.ib()
    envparameters: EnvSettingDefault = attr.ib()
    trainingparameters: TrainingDefault = attr.ib()
    seedparameters: SeedSetting = attr.ib()
    algo: str = "DQN"


def create_dummy_config(CONFIG):
    return copy.deepcopy(CONFIG)


# def save_config(args, save_path, file_name):
#     file = open(os.path.join(str(save_path), str(file_name) + '.yaml'), mode='w', encoding='utf-8')
#     yaml.dump(args.as_dict(), file)
#     file.close()
#
#
# def load_config(config_path: str) -> Dict[str, Any]:
#     with open(config_path) as data_file:
#         return _load_config(data_file)
#
#
# def _load_config(data_file):
#     return yaml.safe_load(data_file)


def load_config2(log_path):
    file = open(os.path.join(str(log_path)), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    print("!!", config_dict)
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
    print("dummy_dict: ", dummy_dict)
    return dummy_dict


if __name__ == "__main__":
    # -----------------------------------
    DQN_CONFIG = TrainerSettings(
        algo="dqn",
        hyperparameters=DQNSettings(),
        envparameters=EnvSettingDefault(),
        trainingparameters=TrainingDefault(),
        seedparameters=SeedSetting())

    # 打印当前参数
    print("----------------------------- 打印当前参数 --------------------------------")
    args = create_dummy_config(DQN_CONFIG)
    print(args)

    save = False
    # ----------------------------------------------------------------
    # 保存参数
    if save:
        print("----------------------------- 保存当前参数 --------------------------------")
        save_path = os.getcwd()
        save_config(args, save_path, file_name="config")

    # ----------------------------------------------------------------
    # load 参数
    else:
        print("----------------------------- load 当前参数 --------------------------------")
        config_path = "config.yaml"
        # Load YAML
        configured_dict: Dict[str, Any] = {
            "hyperparameters": {},
            "envparameters": {},
            "trainingparameters": {},
            "seedparameters": {},
            "agent_name": {},
            "env_name":{}

        }
        _require_all_behaviors = True
        if config_path is not None:
            configured_dict.update(load_config(config_path))
        else:
            # If we're not loading from a file, we don't require all behavior names to be specified.
            _require_all_behaviors = False
        print("configured_dict: ", configured_dict)
        # print(cattr.unstructure(TrainerSettings))
        print(cattr.structure(configured_dict, TrainerSettings))
        print("-------------------------------------------------------------------------")
        my_args = load_config2("config.yaml")
        print("args: ", my_args)
        print("max_steps: ", my_args.max_episodes)


