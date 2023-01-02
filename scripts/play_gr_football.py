from rollout.rollout_func import rollout_func
from utils.desc.task_desc import RolloutDesc
from envs.gr_football.env import GRFootballEnv
from algorithm.mappo.policy import MAPPO
from utils.cfg import load_cfg
import numpy as np
import pickle as pkl

config_path = "/home/yansong/Desktop/jidiai/ai_lib/expr/gr_football/expr_imitation_psro.yaml"
model_path_0 = "/home/yansong/Desktop/jidiai/ai_lib/trained_models/gr_football/11_vs_11/built_in"
model_path_1 = "/home/yansong/Desktop/jidiai/ai_lib/trained_models/gr_football/11_vs_11/built_in"

cfg = load_cfg(config_path)

policy_id_0 = "policy_0"
policy_id_1 = "policy_1"
policy_0 = MAPPO.load(model_path_0, env_agent_id="agent_0")
policy_1 = MAPPO.load(model_path_1, env_agent_id="agent_1")

env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
rollout_desc = RolloutDesc("agent_0", None, None, None, None, None, None)
behavior_policies = {
    "agent_0": (policy_id_0, policy_0),
    "agent_1": (policy_id_1, policy_1),
}

rollout_results = rollout_func(
    eval=True,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=None,
    rollout_length=3001,
    render=False,
)

print(rollout_results)
