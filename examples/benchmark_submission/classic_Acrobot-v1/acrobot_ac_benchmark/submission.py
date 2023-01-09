from pathlib import Path
import os
import torch
from torch.distributions import Categorical


current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "policy_1000.pth")


STATE_DIM = 6
ACTION_DIM = 3
HIDDEN_SIZE = 64
NUM_HIDDEN_LAYER = 1


import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_inputs,
                 num_outputs,
                 hidden_size,
                 num_hidden_layer=0):
        super(ActorCritic, self).__init__()

        # self.critic = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )
        critic_linear_in = nn.Linear(num_inputs, hidden_size)
        critic_linear_out = nn.Linear(hidden_size, 1)
        critic_hidden_list = []
        if num_hidden_layer > 0:
            for _ in range(num_hidden_layer):
                critic_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                critic_hidden_list.append(nn.ReLU())
        self.critic = nn.Sequential(
            critic_linear_in, nn.ReLU(), *critic_hidden_list, critic_linear_out
        )

        actor_linear_in = nn.Linear(num_inputs, hidden_size)
        actor_linear_out = nn.Linear(hidden_size, num_outputs)
        actor_hidden_list = []
        if num_hidden_layer > 0:
            for _ in range(num_hidden_layer):
                actor_hidden_list.append(nn.Linear(hidden_size, hidden_size))
                actor_hidden_list.append(nn.ReLU())
        self.actor = nn.Sequential(
            actor_linear_in,
            nn.ReLU(),
            *actor_hidden_list,
            actor_linear_out,
            nn.Softmax(dim=1)
        )

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



policy = ActorCritic(num_inputs=STATE_DIM,
                     num_outputs=ACTION_DIM,
                     hidden_size=HIDDEN_SIZE,
                     num_hidden_layer=NUM_HIDDEN_LAYER)

policy.load_state_dict(torch.load(model_path))

def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']).unsqueeze(0)
    probs, value = policy(obs)
    action = torch.argmax(probs)
    onehot_a = [0]*ACTION_DIM
    onehot_a[action.item()] = 1
    return [onehot_a]
