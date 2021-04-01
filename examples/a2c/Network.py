import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist

class Critic(nn.Module):
    def __init__(self, state_space, output_size, hidden_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        value = self.critic(x)
        return value