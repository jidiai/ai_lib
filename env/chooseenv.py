# -*- coding:utf-8  -*-
from env.sokoban import Sokoban
from env.snakes import SnakeEatBeans
from env.reversi import Reversi
from env.gobang import GoBang
from env.ccgame import CCGame
import gym
import configparser
import os


def make(env_type, conf=None):
    config = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(path, encoding="utf-8")
    env_list = config.sections()
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
    elif name == "classic":
        env_core = gym.make(env_type.split('_')[1])
        env = CCGame(conf_dic[env_type], env_core)

    return env


