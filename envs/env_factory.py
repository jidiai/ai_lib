from registry import registry

MAX_ENV_PER_WORKER=100

def make_envs(worker_id,worker_seed,cfg):
    envs={}
    assert len(cfg)<MAX_ENV_PER_WORKER
    for idx,env_cfg in enumerate(cfg):
        env_name=env_cfg["cls"]
        id_prefix=env_cfg["id_prefix"]
        env_id="{}_{}_{}".format(worker_id,id_prefix,idx)
        assert env_id not in envs
        env_seed=worker_seed*MAX_ENV_PER_WORKER+idx
        env_cls=registry.get(registry.ENV,env_name)
        env=env_cls(env_id,env_seed,env_cfg)
        envs[env_id]=env
    return envs