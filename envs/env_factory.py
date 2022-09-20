from random import seed
from .gr_football.env import GRFootballEnv
from .kuhn_poker.env import KuhnPokerEnv
from .connect_four.env import ConnectFourEnv
from .leduc_poker.env import LeducPokerEnv

MAX_ENV_PER_WORKER=100

def make_gr_football_env(env_id,env_seed,env_cfg):
    env=GRFootballEnv(env_id,env_seed,env_cfg)
    return env

def make_kuhn_poker_env(env_id,env_seed,env_cfg):
    env=KuhnPokerEnv(env_id,env_seed,env_cfg)
    return env

def make_connect_four_env(env_id,env_seed,env_cfg):
    env=ConnectFourEnv(env_id,env_seed,env_cfg)
    return env

def make_leduc_poker_env(env_id,env_seed,env_cfg):
    env=LeducPokerEnv(env_id,env_seed,env_cfg)
    return env

def make_envs(worker_id,worker_seed,cfg):
    envs={}
    assert len(cfg)<MAX_ENV_PER_WORKER
    for idx,env_cfg in enumerate(cfg):
        cls=env_cfg["cls"]
        id_prefix=env_cfg["id_prefix"]
        env_id="{}_{}_{}".format(worker_id,id_prefix,idx)
        assert env_id not in envs
        env_seed=worker_seed*MAX_ENV_PER_WORKER+idx
        func="make_{}_env".format(cls)
        env=globals()[func](env_id,env_seed,env_cfg)
        envs[env_id]=env
    return envs