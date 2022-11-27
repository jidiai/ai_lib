import argparse
from cfg import load_cfg
from envs.env_factory import make_kuhn_poker_env
from envs.kuhn_poker.env import KuhnPokerEnv
import numpy as np

from utils.episode import EpisodeKey


class CMDPolicy:
    def __init__(self):
        pass

    def compute_action(self, *args):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    env_cfg = cfg.rollout_manager.worker.envs[0]
    assert env_cfg.cls == "kuhn_poker"
    env: KuhnPokerEnv = make_kuhn_poker_env(0, 0, env_cfg)

    env_rets = env.reset(None)
    while not env.is_terminated():
        print("step", env.step_ctr)
        print("env_rets", env_rets)
        curr_agent_ids = env.get_current_agent_ids()
        assert len(curr_agent_ids) == 1
        agent_id = curr_agent_ids[0]
        action_mask = env_rets[agent_id][EpisodeKey.ACTION_MASK]
        probs = action_mask / np.sum(action_mask)
        actions = {
            agent_id: np.random.choice([0, 1], p=probs) for agent_id in curr_agent_ids
        }
        print("actions", actions)
        env_rets = env.step(actions)
    print(env_rets)


if __name__ == "__main__":
    main()
