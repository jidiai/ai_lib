import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.affine1 = nn.Linear(self.input_size, 128)
        self.affine2 = nn.Linear(128, self.output_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class NoisyActor(nn.Module):
    """
    continuous actor with random noise
    """
    def __init__(self, state_dim, hidden_dim, out_dim, num_hidden_layer=0, tanh=False, action_high = 1, action_low = -1):
        super(NoisyActor, self).__init__()

        self.linear_in = nn.Linear(state_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.num_hidden_layer = num_hidden_layer

        if self.num_hidden_layer > 0:
                hid_net = []
                for _ in range(self.num_hidden_layer):
                    hid_net.append(nn.Linear(hidden_dim, hidden_dim))
                    hid_net.append(nn.ReLU())
                self.linear_hid = nn.Sequential(*hid_net)

        self.apply(weights_init_)
        self.noise = torch.Tensor(1)
        self.tanh = tanh
        if tanh:  #normalise the action
            self.action_scale = torch.FloatTensor([(action_high - action_low) / 2.])
            self.action_bias = torch.FloatTensor([(action_high + action_low) / 2.])

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        if self.tanh:
            mean = torch.tanh(x) * self.action_scale + self.action_bias
        else:
            mean = x
        return mean

    def sample(self, state):
        """
        :return: (sampled_action, prob, logprob, mean)
        """
        mean = self.forward(state)
        noise = self.noise.normal_(0., std = 0.1)    #all these hyperparameters can be defined in advance
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(1.), torch.tensor(0.), mean


class CategoricalActor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CategoricalActor, self).__init__()

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

class openai_actor(nn.Module):
    def __init__(self, num_inputs, action_size):
        super(openai_actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, 128)
        self.linear_a2 = nn.Linear(128, 64)
        self.linear_a = nn.Linear(64, action_size)
        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
    
    def forward(self, input, original_out=False):
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if original_out == True:   return model_out, policy
        return policy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class GaussianActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, tanh=False, action_high = 2, action_low = -2):
        super(GaussianActor, self).__init__()

        self.linear_in = nn.Linear(state_dim, hidden_dim)
        self.linear_hid = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.logstd_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        self.tanh = tanh
        if tanh:  # normalise the action
            self.action_scale = torch.FloatTensor([(action_high - action_low) / 2.])
            self.action_bias = torch.FloatTensor([(action_high + action_low) / 2.])

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        x = F.relu(self.linear_hid(x))
        mean = self.mean_linear(x)
        log_std = self.logstd_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, logstd = self.forward(state)
        std = logstd.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        if self.tanh:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(1, keepdim = True)
            mean = mean

        return action, log_prob, mean
