from algo.ppo.ppo import PPO
from algo.dqn.dqn import DQN
from algo.ddqn.ddqn import DDQN
from algo.pg.pg import PG
# todo: random


def ini_agents(args):
    # if args.algo == "ppo":
    #     agent = PPO(args)
    agent = None
    if args.algo == "dqn":
        agent = DQN(args)
    # elif args.algo == "ppo":
    #     agent = PPO(args)
    elif args.algo == "pg":
        agent = PG(args)
    elif args.algo == "ddqn":
        agent = DDQN(args)
    return agent


class BaseAgent:
    def __init__(self, state_dim, action_dim, args):
        pass

    def inference(self, observation, train=True):
        result = self.agent.choose_action
        pass

    def store_transition(self, transition):
        pass

    def learn(self):
        pass

    def save(self):
        pass

    def load(self, file):
        pass

