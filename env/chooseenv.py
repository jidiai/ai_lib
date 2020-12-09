# -*- coding:utf-8  -*-
from .sokoban import Sokoban
from .snakes import SnakeEatBeans
from .reversi import Reversi
from .gobang import GoBang
import configparser
import os


def make(env_type, conf=None):
    config = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), 'config.ini')
    # print(path)
    config.read(path)
    env_list = ["gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_5p"]
    conf_dic = {}
    for env_name in env_list:
        conf_dic[env_name] = config[env_name]
    if env_type not in env_list:
        raise Exception("可选环境列表：%s,传入环境为%s" % (str(env_list), env_type))
    if conf:
        conf_dic[env_type] = conf

    name = env_type.split('_')[0]

    if name == "gobang":
        env = GoBang(conf_dic[env_type])
    elif name == "reversi":
        env = Reversi(conf_dic[env_type])
    elif name == "snakes":
        env = SnakeEatBeans(conf_dic[env_type])
    elif name == "sokoban":
        env = Sokoban(conf_dic[env_type])

    return env




