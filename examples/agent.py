# from algo.ppo.ppo import PPO
from algo.dqn.dqn_v3 import DQN

# todo: random

def ini_agents(args):
    # if args.algo == "ppo":
    #     agent = PPO(args)
    agent = None
    if args.algo == "dqn":
        print('@@@@')
        agent = DQN(args)

    return agent

# class BaseAgent:
#     def __init__(self, state_dim, action_dim, args):
#         pass
#
#     def choose_action(self, observation, train=True):
#         pass
#
#     def store_transition(self, transition):
#         pass
#
#     def learn(self):
#         pass
#
#     def save(self):
#         pass
#
#     def load(self, file):
#         pass

