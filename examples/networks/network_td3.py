import torch
import torch.nn as nn
import torch.nn.functional as F


# class old_Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(old_Actor, self).__init__()
#
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_dim)
#
#         self.max_action = max_action
#
#     def forward(self, state):
#         a = F.relu(self.fc1(state))
#         a = F.relu(self.fc2(a))
#         a = torch.tanh(self.fc3(a)) * self.max_action
#         return a

class ContinuousTanhActor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_size, action_loc, action_scale,
                 num_hidden_layer=1):
        super(ContinuousTanhActor, self).__init__()

        self.input_size = state_dim
        self.hidden_size = hidden_size
        self.output_size = action_dim

        actor_linear_in = nn.Linear(self.input_size, self.hidden_size)
        actor_linear_out = nn.Linear(self.hidden_size, action_dim)
        actor_hidden_list = []
        if num_hidden_layer > 0:
            for _ in range(num_hidden_layer):
                actor_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                actor_hidden_list.append(nn.ReLU())
        self.net = nn.Sequential(
            actor_linear_in,
            nn.ReLU(),
            *actor_hidden_list,
            actor_linear_out,
            nn.Tanh()
        )
        self.action_loc = action_loc
        self.action_scale = action_scale
    def forward(self, x):
        x = self.net(x)
        return x*self.action_scale+self.action_loc


# class old_Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(old_Critic, self).__init__()
#
#         self.fc1 = nn.Linear(state_dim + action_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)
#
#     def forward(self, state, action):
#         state_action = torch.cat([state, action], 1)
#
#         q = F.relu(self.fc1(state_action))
#         q = F.relu(self.fc2(q))
#         q = self.fc3(q)
#         return q

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_hidden_layer=1):
        super(Critic, self).__init__()

        self.input_size = state_dim + action_dim
        self.hidden_size = hidden_size
        self.output_size = action_dim

        critic_linear_in = nn.Linear(self.input_size, self.hidden_size)
        critic_linear_out = nn.Linear(self.hidden_size, self.output_size)
        critic_hidden_list = []
        if num_hidden_layer > 0:
            for _ in range(num_hidden_layer):
                critic_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                critic_hidden_list.append(nn.ReLU())
        self.net = nn.Sequential(
            critic_linear_in, nn.ReLU(), *critic_hidden_list, critic_linear_out
        )
    def forward(self, x, u):
        x = self.net(torch.cat([x, u], 1))
        return x


