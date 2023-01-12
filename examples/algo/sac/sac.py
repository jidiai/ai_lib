import sys
from pathlib import Path
import os
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from examples.networks.critic import Critic
from examples.networks.actor import NoisyActor, CategoricalActor, GaussianActor
from examples.networks.encoder import CNNEncoder, Flatten


base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from examples.common.buffer import Replay_buffer as buffer


def get_trajectory_property():  # for adding terms to the memory buffer
    return ["action"]


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def update_params(optim, loss, clip=False, param_list=[], retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if clip is not False:
        for i in param_list:
            torch.nn.utils.clip_grad_norm_(i, clip)
    optim.step()


class SAC(object):
    def __init__(self, args):

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.gamma = args.gamma
        self.tau = args.tau

        self.action_continuous = args.action_continuous

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.alpha_lr = args.alpha_lr
        self.use_cuda = args.use_cuda
        if 'max_grad_norm' not in vars(args):
            self.max_grad_norm = None
        else:
            self.max_grad_norm = args.max_grad_norm

        self.buffer_size = args.buffer_capacity

        self.policy_type = (
            "discrete" if (not self.action_continuous) else args.policy_type
        )  # deterministic or gaussian policy
        self.device = "cuda" if self.use_cuda else 'cpu'

        given_critic = Critic  # need to set a default value
        self.preset_alpha = args.alpha

        if self.policy_type == "deterministic":
            self.tune_entropy = False
            hid_layer = args.num_hid_layer

            self.policy = NoisyActor(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_size,
                out_dim=1,
                num_hidden_layer=hid_layer,
            ).to(self.device)
            self.policy_target = NoisyActor(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_size,
                out_dim=1,
                num_hidden_layer=hid_layer,
            ).to(self.device)
            self.policy_target.load_state_dict(self.policy.state_dict())

            self.q1 = given_critic(
                self.state_dim + self.action_dim,
                self.action_dim,
                self.hidden_size,
                hid_layer,
            ).to(self.device)
            self.q1.apply(weights_init_)

            self.q1_target = given_critic(
                self.state_dim + self.action_dim,
                self.action_dim,
                self.hidden_size,
                hid_layer,
            ).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.critic_optim = Adam(self.q1.parameters(), lr=self.critic_lr)

        elif self.policy_type == "discrete":
            self.tune_entropy = args.tune_entropy
            self.target_entropy_ratio = args.target_entropy_ratio

            if 'cnn' in vars(args):
                self.use_cnn_encoder = True
                cnn_cfg = args.cnn
                # self.cnn_encoder = Flatten()
                self.actor_cnn_encoder = CNNEncoder(input_chanel=cnn_cfg['input_chanel'],
                                                    hidden_size=None,
                                                    output_size=None,
                                                    channel_list=cnn_cfg['channel_list'],
                                                    kernel_list=cnn_cfg['kernel_list'],
                                                    stride_list=cnn_cfg['stride_list'],
                                                    batch_norm=False).to(self.device)
                self.critic_cnn_encoder = CNNEncoder(input_chanel=cnn_cfg['input_chanel'],
                                                     hidden_size=None,
                                                     output_size=None,
                                                     channel_list=cnn_cfg['channel_list'],
                                                     kernel_list=cnn_cfg['kernel_list'],
                                                     stride_list=cnn_cfg['stride_list'],
                                                     batch_norm=False).to(self.device)
                self.state_dim = self.hidden_size
            else:
                self.use_cnn_encoder = False

            self.policy = CategoricalActor(
                self.state_dim, self.hidden_size, self.action_dim
            ).to(self.device)

            hid_layer = args.num_hidden_layer
            self.q1 = given_critic(
                self.state_dim, self.action_dim, self.hidden_size, hid_layer
            ).to(self.device)
            self.q1.apply(weights_init_)
            self.q2 = given_critic(
                self.state_dim, self.action_dim, self.hidden_size, hid_layer
            ).to(self.device)
            self.q2.apply(weights_init_)

            self.q1_target = given_critic(
                self.state_dim, self.action_dim, self.hidden_size, hid_layer
            ).to(self.device)
            self.q2_target = given_critic(
                self.state_dim, self.action_dim, self.hidden_size, hid_layer
            ).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

            if self.use_cnn_encoder:
                self.critic_optim = Adam(list(self.q1.parameters())+
                                         list(self.q2.parameters())+
                                         list(self.critic_cnn_encoder.parameters()),
                                         lr=self.critic_lr)
            else:
                self.critic_optim = Adam(
                    list(self.q1.parameters()) + list(self.q2.parameters()),
                    lr=self.critic_lr,
                )

        elif self.policy_type == "gaussian":
            self.tune_entropy = args.tune_entropy
            self.target_entropy_ratio = args.target_entropy_ratio

            self.policy = GaussianActor(
                self.state_dim, self.hidden_size, 1, tanh=False
            ).to(self.device)
            # self.policy_target = GaussianActor(self.state_dim, self.hidden_size, 1, tanh = False).to(self.device)

            hid_layer = args.num_hid_layer
            self.q1 = given_critic(
                self.state_dim + self.action_dim,
                self.action_dim,
                self.hidden_size,
                hid_layer,
            ).to(self.device)
            self.q1.apply(weights_init_)
            self.critic_optim = Adam(self.q1.parameters(), lr=self.critic_lr)

            self.q1_target = given_critic(
                self.state_dim + self.action_dim,
                self.action_dim,
                self.hidden_size,
                hid_layer,
            ).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())

        else:
            raise NotImplementedError

        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)

        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        if self.use_cnn_encoder:
            self.policy_optim = Adam(list(self.policy.parameters())+
                                     list(self.actor_cnn_encoder.parameters()),
                                     lr=self.actor_lr)
        else:
            self.policy_optim = Adam(self.policy.parameters(), lr=self.actor_lr)

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        if self.tune_entropy:
            self.target_entropy = (
                -np.log(1.0 / self.action_dim) * self.target_entropy_ratio
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            # self.alpha = self.log_alpha.exp()
            self.alpha = torch.tensor([self.preset_alpha]).to(self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(
                [self.preset_alpha]
            ).to(self.device)  # coefficiency for entropy term

    def choose_action(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        if self.policy_type == "discrete":
            if self.use_cnn_encoder:
                state = self.actor_cnn_encoder(state.unsqueeze(0))
            else:
                state = state.view(1,-1)

            if train:
                action, _, _, _ = self.policy.sample(state)
                action = action.item()
                self.add_experience({"action": action})
            else:
                _, _, _, action = self.policy.sample(state)
                action = action.item()
            return {"action": action}

        elif self.policy_type == "deterministic":
            state = state.view(1, -1)
            if train:
                _, _, _, action = self.policy.sample(state)
                action = action.item()
                self.add_experience({"action": action})
            else:
                _, _, _, action = self.policy.sample(state)
                action = action.item()
            return {"action": action}

        elif self.policy_type == "gaussian":
            state = state.view(1, -1)
            if train:
                action, _, _ = self.policy.sample(state)
                action = action.detach().numpy().squeeze(1)
                self.add_experience({"action": action})
            else:
                _, _, action = self.policy.sample(state)
                action = action.detach().numpy().squeeze(1)
            return {"action": action}

        else:
            raise NotImplementedError

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def critic_loss(self, current_state, batch_action, next_state, reward, mask):

        if self.use_cnn_encoder:
            critic_encoded_state = self.critic_cnn_encoder(current_state)
            critic_encoded_next_state = self.critic_cnn_encoder(next_state)
            actor_encoded_next_state = self.actor_cnn_encoder(next_state)
        else:
            critic_encoded_state = current_state
            critic_encoded_next_state = next_state
            actor_encoded_next_state = next_state

        with torch.no_grad():
            next_state_action, next_state_pi, next_state_log_pi, _ = self.policy.sample(
                actor_encoded_next_state
            )
            # qf1_next_target, qf2_next_target = self.critic_target(next_state)
            qf1_next_target = self.q1_target(critic_encoded_next_state)
            qf2_next_target = self.q2_target(critic_encoded_next_state)

            min_qf_next_target = next_state_pi * (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )  # V function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        # qf1, qf2 = self.critic(current_state)  # Two Q-functions to mitigate positive bias in the policy improvement step, [batch, action_num]
        qf1 = self.q1(critic_encoded_state)
        qf2 = self.q2(critic_encoded_state)

        qf1 = qf1.gather(1, batch_action.long())
        qf2 = qf2.gather(
            1, batch_action.long()
        )  # [batch, 1] , pick the actin-value for the given batched actions

        qf1_loss = torch.mean((qf1 - next_q_value).pow(2))
        qf2_loss = torch.mean((qf2 - next_q_value).pow(2))

        return qf1_loss, qf2_loss

    def policy_loss(self, current_state):
        if self.use_cnn_encoder:
            critic_encoded_state = self.critic_cnn_encoder(current_state)
            actor_encoded_state = self.actor_cnn_encoder(current_state)
        else:
            critic_encoded_state = current_state
            actor_encoded_state = current_state

        with torch.no_grad():
            # qf1_pi, qf2_pi = self.critic(current_state)
            qf1_pi = self.q1(critic_encoded_state)
            qf2_pi = self.q2(critic_encoded_state)

            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        pi, prob, log_pi, _ = self.policy.sample(actor_encoded_state)
        inside_term = self.alpha.detach() * log_pi - min_qf_pi  # [batch, action_dim]
        policy_loss = ((prob * inside_term).sum(1)).mean()

        return policy_loss, prob.detach(), log_pi.detach()

    def alpha_loss(self, action_prob, action_logprob):

        if self.tune_entropy:
            entropies = -torch.sum(
                action_prob * action_logprob, dim=1, keepdim=True
            )  # [batch, 1]
            entropies = entropies.detach()
            alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies))

            alpha_logs = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_logs = self.alpha.detach().clone()

        return alpha_loss, alpha_logs

    def learn(self):

        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return {}

        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data["states"]),
            "o_next_0": np.array(data["states_next"]),
            "r_0": np.array(data["rewards"]).reshape(-1, 1),
            "u_0": np.array(data["action"]),
            "d_0": np.array(data["dones"]).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float).to(self.device)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float).to(self.device)
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(
            self.batch_size, -1
        ).to(self.device)
        reward = torch.tensor(transitions["r_0"], dtype=torch.float).to(self.device)
        done = torch.tensor(transitions["d_0"], dtype=torch.float).to(self.device)

        # if self.use_cnn_encoder:
        #     critic_encoded_obs = self.critic_cnn_encoder(obs)
        #     critic_encoded_obs_ = self.critic_cnn_encoder(obs_)
        #     actor_encoded_obs = self.actor_cnn_encoder(obs)
        # else:
        #     critic_encoded_obs = obs
        #     critic_encoded_obs_ = obs_
        #     actor_encoded_obs = obs

        if self.policy_type == "discrete":
            qf1_loss, qf2_loss = self.critic_loss(obs, action, obs_, reward, (1 - done))
            policy_loss, prob, log_pi = self.policy_loss(obs)
            alpha_loss, alpha_logs = self.alpha_loss(prob, log_pi)
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
                if self.use_cnn_encoder:
                    torch.nn.utils.clip_grad_norm_(self.critic_cnn_encoder.parameters(), self.max_grad_norm)
            grad_dict = {}
            for name, param in self.q1.named_parameters():
                grad_dict[f"Q1/{name} gradient"] = param.grad.mean().item()
            if self.use_cnn_encoder:
                for name, param in self.critic_cnn_encoder.named_parameters():
                    grad_dict[f'Critic_CNN_encoder/{name} gradient'] = param.grad.mean().item()

            self.critic_optim.step()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if self.use_cnn_encoder:
                    torch.nn.utils.clip_grad_norm_(self.actor_cnn_encoder.parameters(), self.max_grad_norm)

            for name, param in self.policy.named_parameters():
                grad_dict[f"Policy/{name} gradient"] = param.grad.mean().item()
            if self.use_cnn_encoder:
                for name, param in self.actor_cnn_encoder.named_parameters():
                    grad_dict[f'Actor_CNN_encoder/{name} gradient'] = param.grad.mean().item()

            self.policy_optim.step()
            # update_params(self.critic_optim, qf_loss, clip = self.max_grad_norm,
            #               param_list=list(self.critic_cnn_encoder.parameters())
            #                          +list(self.q1.parameters())+list(self.q2.parameters()))
            # update_params(self.policy_optim, policy_loss, clip=self.max_grad_norm,
            #               param_list=list(self.actor_cnn_encoder.parameters())+list(self.policy.parameters()))

            if self.tune_entropy:
                update_params(self.alpha_optim, alpha_loss)
                self.alpha = self.log_alpha.exp().detach()

            if self.learn_step_counter % self.target_replace_iter == 0:
                # self.critic_target.load_state_dict(self.critic.state_dict())
                self.q1_target.load_state_dict(self.q1.state_dict())
                self.q2_target.load_state_dict(self.q2.state_dict())

            self.learn_step_counter += 1

            training_results = {
                "policy_loss": policy_loss.detach().cpu().numpy(),
                "value_loss": qf_loss.detach().cpu().numpy(),
                "alpha_loss": alpha_loss.detach().cpu().numpy(),
            }
            training_results.update(grad_dict)

            return training_results

        elif self.policy_type == "deterministic":

            current_q = self.q1(torch.cat([obs, action], 1))

            target_next_action = self.policy_target(obs_)

            target_next_q = self.q1_target(torch.cat([obs_, target_next_action], 1))
            next_q_value = reward + (1 - done) * self.gamma * target_next_q

            qf_loss = F.mse_loss(current_q, next_q_value.detach())
            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            _, _, _, current_action = self.policy.sample(obs)
            qf_pi = self.q1(torch.cat([obs, current_action], 1))
            policy_loss = -qf_pi.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.learn_step_counter % self.target_replace_iter == 0:
                for param, target_param in zip(
                    self.q1.parameters(), self.q1_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.policy.parameters(), self.policy_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

        elif self.policy_type == "gaussian":

            action = torch.tensor(transitions["u_0"], dtype=torch.float).view(
                self.batch_size, -1
            )

            with torch.no_grad():
                # next_action, next_action_logprob, _ = self.policy_target.sample(obs_)
                next_action, next_action_logprob, _ = self.policy.sample(obs_)
                target_next_q = (
                    self.q1_target(torch.cat([obs_, next_action], 1))
                    - self.alpha * next_action_logprob
                )
                next_q_value = reward + (1 - done) * self.gamma * target_next_q
            qf1 = self.q1(torch.cat([obs, action], 1))
            qf_loss = F.mse_loss(qf1, next_q_value)

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _ = self.policy.sample(obs)
            qf_pi = self.q1(torch.cat([obs, pi], 1))
            policy_loss = ((self.alpha * log_pi) - qf_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.tune_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_pi + self.target_entropy).detach()
                ).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
            else:
                pass

            if self.learn_step_counter % self.target_replace_iter == 0:
                for param, target_param in zip(
                    self.q1.parameters(), self.q1_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )
                # for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                #    target_param.data.copy_(self.tau * param.data + (1.-self.tau) * target_param.data)

            training_results = {
                "policy_loss": policy_loss.item(),
                "value_loss": qf_loss.item(),
                "alpha_loss": alpha_loss.item()
            }
            return training_results

        else:
            raise NotImplementedError

    def save(self, save_path, episode):

        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_actor_path)
        if self.use_cnn_encoder:
            encoder_path = os.path.join(base_path, f"encoder_{episode}.pth")
            encoder_state_dict = self.actor_cnn_encoder.state_dict()
            torch.save(encoder_state_dict, encoder_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))
