import importlib
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from agents.baseagent import Baseagent


def ini_agents(args):
    agent_file_name = str("algo." + str(args.algo) + "." + str(args.algo))
    agent_file_import = importlib.import_module(agent_file_name)
    agent_class_name = args.algo.upper()

    # 实例化agent
    agent = getattr(agent_file_import, agent_class_name)(args)
    return agent


class SingleRLAgent(Baseagent):
    def __init__(self, args):
        super(SingleRLAgent, self).__init__(args)
        self.args = args
        self.algo = ini_agents(args)
        self.set_agent()

    def set_agent(self):
        self.agent.append(self.algo)

    def action_from_algo_to_env(self, joint_action):
        '''
        :param joint_action:
        :return: wrapped joint action: one-hot
        '''

        joint_action_ = []
        for a in range(self.args.n_player):
            action_a = joint_action["action"]
            if not self.args.action_continuous:   #discrete action space
                each = [0] * self.args.action_space
                each[action_a] = 1
                joint_action_.append(each)
            else:
                joint_action_.append(action_a)   #continuous action space

        return joint_action_



