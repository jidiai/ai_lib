import sys
from pathlib import Path
import os
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


class Discrete_Actor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discrete_Actor, self).__init__()

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

class Determinisitc_Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_high, action_low):
        super(Determinisitc_Actor, self).__init__()

        self.linear_in = nn.Linear(state_dim, hidden_dim)
        self.linear_hid = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
        self.noise = torch.Tensor(1)

        #print('action high = ', action_high)
        #print('action low = ', action_low)
        self.action_scale = torch.FloatTensor([(action_high - action_low) / 2.])
        self.action_bias = torch.FloatTensor([(action_high + action_low) / 2.])

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        x = F.relu(self.linear_hid(x))
        x = self.linear_out(x)
        #mean = torch.tanh(x) * self.action_scale + self.action_bias
        return x #mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std = 0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(1.), torch.tensor(0.), mean



class SAC(object):
    def __init__(self, args, given_critic):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.gamma = args.gamma
        self.tau = args.tau

        self.action_continuous = args.action_continuous
        #self.update_freq = args.update_freq

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.alpha_lr = args.alpha_lr

        self.buffer_size = args.buffer_capacity

        self.policy_type = 'discrete' if (not self.action_continuous) else args.policy_type      #deterministic or gaussian policy
        self.device = 'cpu'

        if self.policy_type == 'deterministic':
            self.preset_alpha = args.alpha
            self.tune_entropy = False

            self.policy = Determinisitc_Actor(self.state_dim, self.hidden_size, args.action_high, args.action_low)

            hid_layer = args.num_hid_layer
            self.q1 = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q2 = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q1.apply(weights_init_)
            self.q2.apply(weights_init_)

            self.q1_target = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q2_target = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            self.critic_optim = Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.critic_lr)

        elif self.policy_type == 'discrete':
            self.preset_alpha = args.alpha
            self.tune_entropy = args.tune_entropy
            self.target_entropy_ratio = args.target_entropy_ratio

            self.policy = Discrete_Actor(self.state_dim, self.hidden_size, self.action_dim).to(self.device)

            hid_layer = args.num_hid_layer
            self.q1 = given_critic(self.state_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q1.apply(weights_init_)
            self.q2 = given_critic(self.state_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q2.apply(weights_init_)

            self.q1_target = given_critic(self.state_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q2_target = given_critic(self.state_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            self.critic_optim = Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.critic_lr)

        else:
            raise NotImplementedError


        #self.policy = Actor(self.state_dim, self.hidden_size, self.action_dim).to(self.device)
        #self.critic = Critic(self.hidden_size, self.state_dim, self.action_dim).to(self.device)


        #self.critic_target = Critic(self.hidden_size, self.state_dim, self.action_dim).to(self.device)
        #self.critic_target.load_state_dict(self.critic.state_dict())
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)


        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        self.policy_optim = Adam(self.policy.parameters(), lr = self.actor_lr)
        #self.critic_optim = Adam(self.critic.parameters(), lr = self.critic_lr)

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

        if self.policy_type == 'discrete':
            if train:
                action, _, _, _ = self.policy.sample(state)
                action = action.item()
                self.add_experience({"action": action})
            else:
                _, _, _, action = self.policy.sample(state)
                action = action.item()
            return {'action': action}

        elif self.policy_type == 'deterministic':
            if train:
                self.eps = max(self.eps_end, self.eps - self.eps_delay)
                if random.random() < self.eps:
                    #action = random.randrange(self.action_dim)
                    action = np.random.uniform(-2,2)
                    self.add_experience({"action": action})
                else:
                    _,_,_,action = self.policy.sample(state)
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
            #qf1_next_target, qf2_next_target = self.critic_target(next_state)
            qf1_next_target = self.q1_target(next_state)
            qf2_next_target = self.q2_target(next_state)

            min_qf_next_target = next_state_pi * (torch.min(qf1_next_target, qf2_next_target) - self.alpha
                                                  * next_state_log_pi)  # V function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        #qf1, qf2 = self.critic(current_state)  # Two Q-functions to mitigate positive bias in the policy improvement step, [batch, action_num]
        qf1 = self.q1(current_state)
        qf2 = self.q2(current_state)

        qf1 = qf1.gather(1, batch_action.long())
        qf2 = qf2.gather(1, batch_action.long())        #[batch, 1] , pick the actin-value for the given batched actions

        qf1_loss = torch.mean((qf1 - next_q_value).pow(2))
        qf2_loss = torch.mean((qf2 - next_q_value).pow(2))

        return qf1_loss, qf2_loss

    def policy_loss(self, current_state):

        with torch.no_grad():
            #qf1_pi, qf2_pi = self.critic(current_state)
            qf1_pi = self.q1(current_state)
            qf2_pi = self.q2(current_state)

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

        if self.policy_type == 'discrete':
            qf1_loss, qf2_loss = self.critic_loss(obs, action, obs_, reward, (1-done))
            policy_loss, prob, log_pi = self.policy_loss(obs)
            alpha_loss, alpha_logs = self.alpha_loss(prob, log_pi)
            qf_loss = qf1_loss + qf2_loss
            update_params(self.critic_optim,qf_loss)
            update_params(self.policy_optim, policy_loss)
            if self.tune_entropy:
                update_params(self.alpha_optim, alpha_loss)

            if self.learn_step_counter % self.target_replace_iter == 0:
                #self.critic_target.load_state_dict(self.critic.state_dict())
                self.q1_target.load_state_dict(self.q1.state_dict())
                self.q2_target.load_state_dict(self.q2.state_dict())

            self.learn_step_counter += 1

        elif self.policy_type == 'deterministic':

            with torch.no_grad():
                _,_,_,next_state_action = self.policy.sample(obs_)
                qf1_next_target = self.q1_target(torch.cat([obs_, next_state_action], 1))
                qf2_next_target = self.q2_target(torch.cat([obs_, next_state_action], 1))
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = reward + (1-done) * self.gamma * min_qf_next_target
            qf1 = self.q1(torch.cat([obs, action], 1))
            qf2 = self.q2(torch.cat([obs, action], 1))
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            _, _, _, current_action = self.policy.sample(obs)
            qf1_pi = self.q1(torch.cat([obs, current_action], 1))
            qf2_pi = self.q2(torch.cat([obs, current_action], 1))
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = (-min_qf_pi).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            alpha_loss, alpha_logs = torch.tensor(0.).to(self.device), self.alpha.detach().clone()
            if self.learn_step_counter % self.target_replace_iter == 0:
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.-self.tau) * target_param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.-self.tau) * target_param.data)

                #self.q1_target.load_state_dict(self.q1.state_dict())
                #self.q2_target.load_state_dict(self.q2.state_dict())

            self.learn_step_counter += 1
        else:
            raise NotImplementedError


    def save(self, save_path, episode):

        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_actor_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))









