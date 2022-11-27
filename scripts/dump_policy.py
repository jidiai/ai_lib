config_fp = "light_malib/expr/gr_football/expr_gail.yaml"
from utils.cfg import load_cfg

cfg = load_cfg(config_fp)

from algorithm.mappo.policy import MAPPO
from gym.spaces import Box

observation_space = None
action_space = Box(-1000, 1000, (19,))
model_config = cfg["populations"][0]["algorithm"]["model_config"]
custom_config = cfg["populations"][0]["algorithm"]["custom_config"]

policy = MAPPO(
    "MAPPO", None, action_space, model_config, custom_config, env_agent_id="team_0"
)

dump_dir = "light_malib/trained_models/11_vs_11/built_in"
import os

os.makedirs(dump_dir, exist_ok=True)
policy.dump(dump_dir)
