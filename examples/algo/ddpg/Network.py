import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_hidden_layer=1):
        super(Actor, self).__init__()

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
        self.net = nn.Sequential(actor_linear_in,
                                 nn.ReLU(),
                                 *actor_hidden_list,
                                 actor_linear_out,
                                 nn.Softmax(dim=1))

        # self.l1 = nn.Linear(self.input_size, self.hidden_size)
        # self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.l3 = nn.Linear(self.hidden_size, action_dim)


    def forward(self, x):
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = torch.softmax(self.l3(x), dim=1)
        x = self.net(x)
        return x


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
        self.net = nn.Sequential(critic_linear_in,
                                 nn.ReLU(),
                                 *critic_hidden_list,
                                 critic_linear_out)

        # self.l1 = nn.Linear(self.input_size, self.hidden_size)
        # self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.l3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, u):
        # x = F.relu(self.l1(torch.cat([x, u], 1)))
        # x = F.relu(self.l2(x))
        # x = self.l3(x)
        x = self.net(torch.cat([x,u], 1))
        return x