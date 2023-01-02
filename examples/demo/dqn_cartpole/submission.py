import argparse
import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from singleagent import SingleRLAgent


parser = argparse.ArgumentParser()
parser.add_argument("-obs_space", default=4, type=int)
parser.add_argument("-action_space", default=2, type=int)
parser.add_argument("-hidden_size", default=64, type=int)
parser.add_argument("-algo", default="dqn", type=str)
parser.add_argument("-network", default="critic", type=str)
parser.add_argument("-n_player", default=1, type=int)
args = parser.parse_args()

agent = SingleRLAgent(args)
critic_net = os.path.dirname(os.path.abspath(__file__)) + "/critic_200.pth"
agent.load(critic_net)

sys.path.pop(-1)  # just for safety


def my_controller(obs_list, action_space_list, obs_space_list):
    action = agent.choose_action_to_env(obs_list)
    return action
