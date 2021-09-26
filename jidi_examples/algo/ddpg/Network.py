import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()

        self.input_size = state_dim
        self.hidden_size = hidden_size
        self.output_size = action_dim

        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, action_dim)


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.softmax(self.l3(x), dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        self.input_size = state_dim + action_dim
        self.hidden_size = hidden_size
        self.output_size = action_dim

        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x