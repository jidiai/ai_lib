from algo.ppo.ppo import PPO
from algo.dqn.dqn import DQN
from algo.pg.pg import PG
from algo.ddpg.ddpg import DDPG
from algo.ac.ac import AC

def ini_agents(args):
    agent = None
    if args.algo == "dqn":
        agent = DQN(args)
    elif args.algo == "pg":
        agent = PG(args)
    elif args.algo == "ddpg":
        agent = DDPG(args)
    elif args.algo == "ac":
        agent = AC(args)
    elif args.algo == "ppo":
        agent = PPO(args)
    return agent

