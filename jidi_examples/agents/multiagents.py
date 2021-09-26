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


class MultiRLAgents(Baseagent):
    def __init__(self, args):
        super(MultiRLAgents, self).__init__(args)
        self.args = args
        self.algo = ini_agents(args)
        self.set_agent()

    def set_agent(self):
        self.agent = self.algo.agents

    def action_from_algo_to_env(self, joint_action, id):
        '''
        :param joint_action:
        :return: wrapped joint action: one-hot
        '''

        joint_action_ = None
        action_a = joint_action["action"]
        if not self.args.action_continuous:   #discrete action space
            each = [0] * self.args.action_space[id]
            each[action_a] = 1
            joint_action_ = each
        else:
            joint_action_ = action_a   #continuous action space

        return joint_action_

    def choose_action_to_env(self, observation, train=True):
        obs_copy = observation.copy()
        obs = obs_copy["obs"]
        agent_id = obs_copy["controlled_player_index"]
        action_from_algo = self.agent[agent_id].choose_action(obs, train)
        action_to_env = self.action_from_algo_to_env({"action": action_from_algo}, agent_id)
        return action_to_env

    def learn(self):
        self.algo.learn()

    def save(self, p_dir, epoch):
        self.algo.save(p_dir, epoch)