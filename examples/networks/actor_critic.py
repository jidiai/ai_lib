import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, num_hidden_layer=0):
        super(ActorCritic, self).__init__()

        # self.critic = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )
        critic_linear_in = nn.Linear(num_inputs, hidden_size)
        critic_linear_out = nn.Linear(hidden_size,1)
        critic_hidden_list = []
        if num_hidden_layer > 0:
            for _ in range(num_hidden_layer):
                critic_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                critic_hidden_list.append(nn.ReLU())
        self.critic = nn.Sequential(critic_linear_in,
                                    nn.ReLU(),
                                    *critic_hidden_list,
                                    critic_linear_out)

        actor_linear_in = nn.Linear(num_inputs, hidden_size)
        actor_linear_out = nn.Linear(hidden_size, num_outputs)
        actor_hidden_list = []
        if num_hidden_layer>0:
            for _ in range(num_hidden_layer):
                actor_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                actor_hidden_list.append(nn.ReLU())
        self.actor = nn.Sequential(actor_linear_in,
                                   nn.ReLU(),
                                   *actor_hidden_list,
                                   actor_linear_out,
                                   nn.Softmax(dim=1))

        # self.actor = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_outputs),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value