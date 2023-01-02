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

    def forward(self, x):
        x = self.net(x)

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_hidden_layer=1):
        super(Critic, self).__init__()

        self.input_size = state_dim
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
    def forward(self, x):
        x = self.net(x)
        return x