import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

from examples.common.buffer import Replay_buffer as buffer
from networks.actor import OpenaiActor as net_a
from networks.critic import OpenaiCritic as net_c

class Agent():
    def __init__(self, input_dim_a, input_dim_c, output_dim, lr_a=0.01, lr_c=0.01, buffer_capacity=1000000):
        self.lr_c = lr_c
        self.lr_a = lr_a

        self.actor_eval = net_a(input_dim_a, output_dim)
        self.actor_target = net_a(input_dim_a, output_dim)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.optimizer_a = optim.Adam(self.actor_eval.parameters(),lr=self.lr_a)

        self.critic_eval = net_c(input_dim_c[0], input_dim_c[1])
        self.critic_target = net_c(input_dim_c[0], input_dim_c[1])
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.optimizer_c = optim.Adam(self.critic_eval.parameters(),lr=self.lr_c)

        self.memory = buffer(buffer_capacity, ["action"])
        self.memory.init_item_buffers()
    def choose_action(self, observation, train=True):
        observation = torch.tensor(observation, dtype=torch.float64)
        if train:
            action = self.actor_eval(observation).detach().numpy()
            self.add_experience({"action": action})
            return action
        else:
            action, _ = self.actor_target(observation, original_out=True)
            action = F.softmax(action, dim=-1)
            return {"action": action.detach().numpy()}

    def add_experience(self, output):
        for k,v in output.items():
            self.memory.insert(k, None, v)


class MADDPG():
    def __init__(self, args):
        self.gamma = args.gamma #0.97
        self.batch_size = args.batch_size #1256
        self.agents = []
        self.n = args.n_player
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.tao = args.tao
        self.args = args
        self.step = 0
        num_agent = args.n_player
        action_dim = args.action_space
        obs_dim = args.obs_space
        for n in range(num_agent):
            in_a = obs_dim[n] #actor输入是自己的observation
            in_c = [sum(obs_dim), sum(action_dim)] #critic的输入是act(cat(obs), cat(action))
            out = action_dim[n] #输出是自己的action_space
            agent = Agent(in_a, in_c, out, self.lr_a, self.lr_c, args.buffer_capacity)
            self.agents.append(agent)

    def learn(self):
        self.step += 1
        if self.step < self.args.start_step or not self.step % self.args.target_replace==0:
            return
        for id in range(self.n):
            seed = np.random.randint(2**31)
            obs, obs_, action, reward, done = [], [], [], [], []
            for agent in self.agents:
                np.random.seed(seed)
                batch = agent.memory.sample(self.batch_size)
                obs.append(torch.tensor(batch['states'], dtype=torch.float64))
                obs_.append(torch.tensor(batch['states_next'], dtype=torch.float64))
                action.append(torch.tensor(batch['action'], dtype=torch.float64))
                #reward.append(torch.tensor([torch.tensor(reward, dtype=torch.float64) for reward in batch['rewards']]))
                reward.append(torch.cat([torch.tensor(reward, dtype=torch.float64) for reward in batch['rewards']],dim=-1).view(self.batch_size,-1))
                done.append(torch.tensor([torch.tensor(done, dtype=torch.float64) for done in batch['dones']]))
            reward =reward[id][:,id]
            done = done[id]

            '''
            obs = torch.tensor(obs, dtype=torch.float64)
            action = torch.tensor(action, dtype=torch.float64)
            reward = torch.tensor(reward, dtype=torch.float64)
            done = torch.tensor(done, dtype=torch.float64)
            obs_ = torch.tensor(obs_, dtype=torch.float64)
            '''
            action_ = []
            for n in range(self.n):
                action_.append(self.agents[n].actor_target(obs_[n]).detach())
            action_ = torch.cat(action_, dim=1).detach()
            obs_ = torch.cat(obs_, dim=1)
            #x = torch.cat(torch.cat(obs, dim=1), torch.cat(action, dim=1), dim=1)
            #x_ = torch.cat((obs_, action_), dim=1)

            agent = self.agents[id]
            y_target = reward + self.gamma * torch.mul((1 - done) , agent.critic_target(obs_,action_).squeeze().detach())
            y_eval = agent.critic_eval(torch.cat(obs, dim=1), torch.cat(action, dim=1)).squeeze()
            loss = nn.MSELoss()(y_eval, y_target)
            agent.optimizer_c.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.critic_eval.parameters(), 0.5)
            agent.optimizer_c.step()

            action_train, policy = agent.actor_eval(obs[id], original_out=True)
            action[id] = policy
            loss_pse = torch.mean(torch.pow(action_train, 2))
            x_train = torch.cat(action,dim=1)
            J = -torch.mean(agent.critic_eval(torch.cat(obs, dim=1), x_train))
            agent.optimizer_a.zero_grad()
          
            (J + 1e-3 * loss_pse).backward()
            nn.utils.clip_grad_norm_(agent.actor_eval.parameters(), 0.5)
            agent.optimizer_a.step()
            print("Loss_q:",loss,"Loss_a:",J)
        for id in range(self.n):
            agent = self.agents[id]
            for p_target, p_eval in zip(agent.actor_target.parameters(), agent.actor_eval.parameters()):
                p_target.data.copy_((1 - self.tao) * p_target.data + self.tao * p_eval.data)
            for p_target, p_eval in zip(agent.critic_target.parameters(), agent.critic_eval.parameters()):
                p_target.data.copy_((1 - self.tao) * p_target.data + self.tao * p_eval.data)

    def choose_action(self, obs, is_train=True):
        joint_action = []
        for n in range(self.n):
            agent = self.agents[n]
            action = agent.choose_action(obs[n], is_train).detach().numpy()
            joint_action.append(action)
        return joint_action

    def save(self,p_dir,epoch):
        para_dict = {0:None ,1:None, 2:None}
        for n in range(self.n):
            agent = self.agents[n]
            para_dict[n] = agent.actor_target.state_dict()
        torch.save(para_dict, str(p_dir)+'/actor_dict_{}.pth'.format(epoch))
'''
class argument():
    def __init__(self):
        self.gamma = 0.97
        self.lr_c = 0.01
        self.lr_a = 0.01
        self.buffer_size = 1e6
        self.batch_size = 1256
        self.tao = 0.01

if __name__ == '__main__':
    args = argument()
'''