import torch
import torch.nn as nn
import torch.nn.functional as F
from SevenPlus.Inference.BasePolicy import BaseModelPolicy, BaseRulePolicy
from SevenPlus.Utils.Adapter.Model import get_net_parameter, set_net_parameter
from SevenPlus.Utils.Validator.Assert import assert_single_agent_net
import numpy as np
from collections import namedtuple
from SevenPlus.Launcher.LauncherConfig import config as g_config


ENV_NUM = g_config.get_setting('reinforcement.parallel.env_num')

# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        #print('### DEBUG', sum(args.obs_shape) + sum(args.action_shape))
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        dim = 1
        if ENV_NUM > 1:
            dim = 2
        #print('### DEBUG', len(state), len(action), type(state[0]), state[0].size(), type(action[0]), action[0].size())
        state = torch.cat(state, dim=dim)
        #print("state size:", state.size())
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=dim)
        #print("action size:", action.size())
        x = torch.cat([state, action], dim=dim)
        #print("after merge:", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

class Args(object):
    def __init__(self):
        # 注意，在捕食者/猎物的环境下，使用MADDPG训练的是捕食者的模型，所以对手（即猎物）只有一个
        # 这里的adversary定义和MPE环境内部的adversary不同，环境内部只定义了good agent和adversary。具体问题请具体分析。
        self.num_adversaries = 1
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.gamma = 0.95
        self.tau = 0.01

class MADDPG:
    def __init__(self, args, agent_id, device):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id).to(device=device)
        self.critic_network = Critic(args).to(device=device)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id).to(device=device)
        self.critic_target_network = Critic(args).to(device=device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)


    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        #for k, v in transitions.items():
        #    print("## DEBUG train", k, v.shape)

        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        #print("## r:", r.size())
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])


        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()
            #print("## q_next", q_next.size())

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
            #print("## target_q", target_q.size())

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        #if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            # save_model
            #pass
        self.train_step += 1

class Agent(object):
    """
    单个Agent的策略，这里的agent就是智能体，一定要和七层中的Agent类区别开来
    """
    def __init__(self, agent_id, args, device):
        self.policy = MADDPG(args, agent_id, device)

    def get_parameter(self):
        return [
            get_net_parameter(self.policy.actor_network),
            get_net_parameter(self.policy.critic_network),
            get_net_parameter(self.policy.actor_target_network),
            get_net_parameter(self.policy.critic_target_network),
        ]

    #def set_parameter(self, parameter:dict, device=None):
    def set_parameter(self, parameter:list, device=None):
        set_net_parameter(self.policy.actor_network, parameter[0], device)
        set_net_parameter(self.policy.critic_network, parameter[1], device)
        set_net_parameter(self.policy.actor_target_network, parameter[2], device)
        set_net_parameter(self.policy.critic_target_network, parameter[3], device)



class MADDPGPolicy(BaseModelPolicy):
    """
    """
    def __init__(self, single_side, side):
        """
        
        Args:
            single_side: bool类型，环境是单势力（False)还是多势力(True)，多势力主要是指红蓝双方博弈（两个势力）
            side: int类型，此模型对应哪一方势力，0表示红方，1表示蓝方
        """
        super(MADDPGPolicy, self).__init__(single_side, side)
        self.agents = None
        

    def init(self, device=None):
        """
        初始化策略内部的网络模型
        """
        env = self.env.env
        args = Args()
        args.n_players = env.n  # 包含敌人的所有玩家个数
        # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
        # 对于simple_tag(捕食者/猎物)来说，默认是1个猎物
        args.n_agents = env.n - args.num_adversaries
        args.obs_shape = [env.observation_space[i].shape[0]
                          for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
        args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
        #print("## DEBUG XXXXXXX", args.action_shape)
        args.high_action = 1
        args.low_action = -1

        self.args = args
        print("## DEBUG init policy", args.__dict__)

        self.agents = []
        for i in range(args.n_agents):
            self.agents.append(Agent(i, args, device))
            

    def get_parameter(self):
        return [p.get_parameter() for p in self.agents]

    def set_parameter(self, net_parameter, device=None):
        for idx, p in enumerate(net_parameter):
            self.agents[idx].set_parameter(p, device)

    def _inference_one(self, idx, obs):
        pi = self.agents[idx].policy.actor_network(obs).squeeze(0)
        #print("## DEBUG inference one", idx, obs.shape, pi.shape, pi)
        return pi

    def inference(self, state_n):
        actions = [self._inference_one(idx, state) for idx, state in enumerate(state_n)]
        return actions


class RandomRulePolicy(BaseRulePolicy):
    def __init__(self, single_side, side):
        super(RandomRulePolicy, self).__init__(single_side, side)

    def inference(self, num=1):
        if self.side == 0:
            num = 3
        actions = []
        for i in range(num):
            actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

        return actions

