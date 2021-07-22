import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, output_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value