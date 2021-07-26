import sys
from pathlib import Path

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np

#from .algo.sac.Network import Actor, Critic

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer

#from .Network import Actor, Critic

def get_trajectory_property():  #for adding terms to the memory buffer
    return ["action"]


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def update_params(optim, loss, clip=False, param_list=False,retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if clip is not False:
        for i in param_list:
            torch.nn.utils.clip_grad_norm_(i, clip)
    optim.step()


class Actor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):

        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)  # should be followed by a softmax layer
        self.apply(weights_init_)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        prob = F.softmax(x, -1)  # [batch_size, action_dim]
        return prob

    def sample(self, state):
        prob = self.forward(state)

        distribution = Categorical(probs=prob)
        sample_action = distribution.sample().unsqueeze(-1)  # [batch, 1]
        z = (prob == 0.0).float() * 1e-8
        logprob = torch.log(prob + z)
        greedy = torch.argmax(prob, dim=-1).unsqueeze(-1)  # 1d tensor

        return sample_action, prob, logprob, greedy

class Critic(nn.Module):


    def __init__(self,hidden_dim, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.apply(weights_init_)

    def forward(self, state):

        x1 = self.q1(state)
        x2 = self.q2(state)
        return x1, x2



class SAC(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.gamma = args.gamma
        self.tau = args.tau

        #self.update_freq = args.update_freq

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.alpha_lr = args.alpha_lr

        self.buffer_size = args.buffer_capacity

        self.preset_alpha = args.alpha
        self.tune_entropy = args.tune_entropy
        self.target_entropy_ratio = args.target_entropy_ratio
        self.device = 'cpu'

        self.policy = Actor(self.state_dim, self.hidden_size, self.action_dim).to(self.device)
        self.critic = Critic(self.hidden_size, self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.hidden_size, self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        self.policy_optim = Adam(self.policy.parameters(), lr = self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.critic_lr)

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        if self.tune_entropy:
            self.target_entropy = -np.log(1./self.action_dim) * self.target_entropy_ratio
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor([self.preset_alpha])  # coefficiency for entropy term


    def choose_action(self, state, train = True):
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        if train:
            action,_,_,_ = self.policy.sample(state)
            action = action.item()
            self.add_experience({"action": action})
        else:
            _,_,_,action = self.policy.sample(state)
            action = action.item()
        return {'action':action}


    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)


    def critic_loss(self, current_state, batch_action, next_state, reward, mask):

        with torch.no_grad():
            next_state_action, next_state_pi, next_state_log_pi, _ = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state)
            min_qf_next_target = next_state_pi * (torch.min(qf1_next_target, qf2_next_target) - self.alpha
                                                  * next_state_log_pi)  # V function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(current_state)  # Two Q-functions to mitigate positive bias in the policy improvement step, [batch, action_num]
        qf1 = qf1.gather(1, batch_action.long())
        qf2 = qf2.gather(1, batch_action.long())        #[batch, 1] , pick the actin-value for the given batched actions

        qf1_loss = torch.mean((qf1 - next_q_value).pow(2))
        qf2_loss = torch.mean((qf2 - next_q_value).pow(2))

        return qf1_loss, qf2_loss

    def policy_loss(self, current_state):

        with torch.no_grad():
            qf1_pi, qf2_pi = self.critic(current_state)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        pi, prob, log_pi, _ = self.policy.sample(current_state)
        inside_term = self.alpha.detach() * log_pi - min_qf_pi  # [batch, action_dim]
        policy_loss = ((prob * inside_term).sum(1)).mean()

        return policy_loss, prob.detach(), log_pi.detach()

    def alpha_loss(self, action_prob, action_logprob):

        if self.tune_entropy:
            entropies = -torch.sum(action_prob * action_logprob, dim=1, keepdim=True)       #[batch, 1]
            entropies = entropies.detach()
            alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies))

            alpha_logs = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_logs = self.alpha.detach().clone()

        return alpha_loss, alpha_logs

    def learn(self):

        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data['states']),
            "o_next_0": np.array(data['states_next']),
            "r_0": np.array(data['rewards']).reshape(-1, 1),
            "u_0": np.array(data['action']),
            "d_0": np.array(data['dones']).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float)
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(transitions["r_0"], dtype=torch.float)
        done = torch.tensor(transitions["d_0"], dtype=torch.float)

        qf1_loss, qf2_loss = self.critic_loss(obs, action, obs_, reward, (1-done))
        policy_loss, prob, log_pi = self.policy_loss(obs)
        alpha_loss, alpha_logs = self.alpha_loss(prob, log_pi)
        qf_loss = qf1_loss + qf2_loss
        update_params(self.critic_optim,qf_loss)
        update_params(self.policy_optim, policy_loss)
        if self.tune_entropy:
            update_params(self.alpha_optim, alpha_loss)

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
        self.learn_step_counter += 1



    def save_models(self):

        save_dict = {}
        save_dict['policy'] = self.policy.state_dict()
        save_dict['critic'] = self.critic.state_dict()

        torch.save(save_dict, 'actor_critic net.pth')


    def load_models(self, file):

        load_dict = torch.load(file)
        self.policy.load_state_dict(load_dict['policy'])
        self.critic.load_state_dict(load_dict['critic'])







