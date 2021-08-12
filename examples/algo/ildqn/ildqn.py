import torch
import torch.nn as nn
from networks.critic import CNN_Critic
from common.buffer import Replay_buffer as buffer
from torch.optim import Adam
import random
import numpy as np
import os

def get_trajectory_property():  #for adding terms to the memory buffer
    return ["action"]

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DQN_IL(object):
    def __init__(self, obs_dim, action_space, buffer_size, batch_size, target_replace, agent_idx,epsilon=1, max_episode=500):
        """
        """
        self.agent_idx = agent_idx   #for reward and mask selection

        self.lr = 0.0003
        self.gamma = 0.95
        # self.single_handle = single_handle
        self.target_replace_iter = 1

        self.obs_dim = obs_dim
        #self.action_space = action_space
        self.action_dim = action_space

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.critic = CNN_Critic(output_size=self.action_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = CNN_Critic(output_size=self.action_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.eps = epsilon
        self.eps_end = 0.05
        self.eps_delay = 1 / (max_episode * 100)  # after 10 round of training

        trajectory_property = get_trajectory_property()
        self.memory = buffer(buffer_size, trajectory_property)
        self.memory.init_item_buffers()
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.target_replace_iter = target_replace

    def choose_action(self, state, train=True):
        """
        state: [13,13,5]
        """
        if state is None:
            #print('this agent has died')
            return 6

        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                #action = np.array([self.action_space.sample()])
                action = np.random.choice(self.action_dim, 1)
            else:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                critic = self.critic(state)
                action = critic.cpu().detach().max(1)[1].numpy()
            self.add_experience({"action": action})
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            critic = self.critic(state)
            action = critic.cpu().detach().max(1)[1].numpy()

        return int(action) #{"action": int(action)}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)


    def learn(self):
        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return

        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data['states']),
            "o_next_0": np.array(data['states_next']),
            "r_0": np.array(data['rewards']).reshape(-1, 1),
            "u_0": np.array(data['action']),
            "d_0": np.array(data['dones']).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float).to(self.device)               #[batch, 13, 13,5]
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float).to(self.device)         #[batch, 13,13,5]
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(self.batch_size, -1).to(self.device)   #[batch,1]
        reward = torch.tensor(transitions["r_0"],dtype=torch.float
                              ).squeeze().reshape(self.batch_size, -1)[:,self.agent_idx].unsqueeze(-1).to(self.device)    #[batch,1]
        done = torch.tensor(transitions["d_0"],dtype=torch.float
                            ).squeeze().reshape(self.batch_size, -1)[:,self.agent_idx].unsqueeze(-1).to(self.device)   #[batch,1]


        batch_current_q = self.critic(obs).gather(1, action.long())  # [batch, 1]
        batch_next_q = self.critic_target(obs_).detach()  # [batch, action_dim]
        batch_next_max_q = batch_next_q.max(1)[0].unsqueeze(-1)  # [batch, 1]
        batch_target = (reward + self.gamma * (1-done) * batch_next_max_q).view(self.batch_size, 1)

        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_current_q, batch_target)

        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
        self.learn_step_counter += 1

        #print('agent {}th trained'.format(self.agent_idx))

        return loss.item()


class Random(object):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def choose_action(self, state, train=True):
        if state is None:
            #print('this agent has died')
            return 6
        else:
            return 6
            #return int(np.random.choice(self.action_dim, 1))

    def learn(self):
        return

class greedy(object):
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.attack_dict = {(5, 5): 13, (6, 5): 14, (7, 5): 15, (5, 6): 16, (7, 6): 17, (5, 7): 18, (6, 7): 19,
                       (7, 7): 20}  # coord
        self.attack_coord = list(self.attack_dict.keys())

    def choose_action(self, state, tarin=True):
        if state is None:
            #print('this agent has died')
            return 6

        state = np.array(state)
        #print('state shape', np.array(state).shape)
        opponent_state = state[:, :, 3]
        action_list = []
        for i in self.attack_coord:
            coord1, coord2 = i
            if opponent_state[coord2, coord1] > 0:
                action_list.append(self.attack_dict[i])
        # print(action_list)
        if len(action_list) == 0:
            return int(np.random.choice(self.action_dim, 1))
        else:
            return int(np.random.choice(action_list, 1))




class ILDQN():
    def __init__(self, args):

        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        trajectory_property = get_trajectory_property()

        self.agents = []
        self.memories = []

        self.single_handle = args.single_handle
        #self.n = args.n_player/2 if args.single_handle else args.n_player
        self.num_agents = args.n_player
        self.num_controlled_agents = self.num_agents/2

        self.obs_space = args.obs_space
        self.action_space = 21 #args.action_space

        for i in range(int(self.num_controlled_agents)):
            self.agents.append(DQN_IL(obs_dim = self.obs_space, action_space = self.action_space,
                                      buffer_size=self.buffer_size, batch_size=self.batch_size,
                                      target_replace=args.target_replace, agent_idx=i))

            #self.agents.append([{'model': DQN_IL(obs_dim = args.obs_space, action_space = args.action_space,
            #                                     buffer_size=self.buffer_size, batch_size=self.batch_size,
            #                                     target_replace=args.target_replace),"controlled_player_index": i}])
            temp_buffer = buffer(self.buffer_size, trajectory_property)
            temp_buffer.init_item_buffers()
            self.memories.append(temp_buffer)
            #self.memories.append([{'buffer': temp_buffer,
            #                       "controlled_player_index": i}])
        for _ in range(int(self.num_agents - self.num_controlled_agents)):
            self.agents.append(Random(action_dim=self.action_space))
            #self.agents.append(greedy(action_dim=self.action_space))


    def learn(self):
        for n in range(self.num_agents):
            agent_model = self.agents[n]
            #_ = agent[0]['model'].learn()
            _ = agent_model.learn()

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")

        para_dict = {0: None, 1: None, 2: None}
        for n in range(int(self.num_controlled_agents)):
            agent = self.agents[n]
            para_dict[n] = agent.critic.state_dict()
        torch.save(para_dict, model_critic_path)

