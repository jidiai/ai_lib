import importlib
from baseagent import Baseagent


def ini_agents(args):
    """
    定义具体算法，例如DQN
    """
    agent_file_name = str(str(args.algo))
    agent_file_import = importlib.import_module(agent_file_name)
    agent_class_name = args.algo.upper()

    network_file_name = str(str(args.network))
    network_file_import = importlib.import_module(network_file_name)
    network_class_name = args.network.capitalize()
    network = getattr(network_file_import, network_class_name)

    # 实例化agent
    agent = getattr(agent_file_import, agent_class_name)(args, network)
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
            each = [0] * self.args.action_space
            each[action_a] = 1
            joint_action_.append(each)
        return joint_action_



