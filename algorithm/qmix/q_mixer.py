
import torch
import torch.nn as nn

from gym.spaces import Discrete

import torch.nn.functional as F

import copy


class QMixer(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
    ):
        super(QMixer, self).__init__()
        if observation_space is None:
            observation_space = 42
        if action_space is None:
            action_space = Discrete(19)
        # super().__init__(
        #     model_config, observation_space, action_space, custom_config, initialization
        # )

        self.n_agent = model_config['n_agent']
        self.embed_dim = model_config['mixer_embed_dim']
        self.hyper_hidden_dim = model_config['hyper_hidden_dim']

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(observation_space, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim*self.n_agent)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(observation_space, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim)
        )

        self.hyper_b_1 = nn.Linear(observation_space, self.embed_dim)

        self.V = nn.Sequential(
            nn.Linear(observation_space, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def to_device(self, device):
        self.to(device)
        # return self_copy

    def forward(self, agent_qs, obs):
        bs = agent_qs.size(0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        agent_qs = torch.as_tensor(agent_qs, dtype=torch.float32)
        agent_qs = agent_qs.view(-1, 1, self.n_agent)
        # First layer
        w1 = torch.abs(self.hyper_w_1(obs))
        b1 = self.hyper_b_1(obs)
        w1 = w1.view(-1, self.n_agent, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(obs))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(obs).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1)
        return q_tot

