import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    """
    output discrete action and logprob of uniform distribution
    num_inputs: input dimension (state_dim)
    num_actions: action_dimension, in our case it should be the number of possible actions, such as 2 in cartpole
    action_space: in discrete case, this can be a list of real action with scales.

    """

    def __init__(self, state_dim , hidden_dim, action_dim):

        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)  # should be followed by a softmax layer
        self.apply(weights_init_)

    def forward(self, state):
        """
        [batch, state_dim]
        """
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
    """
    Double Q function for positive bias, use the minimum Q for gradient update of V function
    in discrete case, we do not take particular action as input, instead we output the value for all actions

    """

    def __init__(self,hidden_dim, state_dim=4, action_dim=2):
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
