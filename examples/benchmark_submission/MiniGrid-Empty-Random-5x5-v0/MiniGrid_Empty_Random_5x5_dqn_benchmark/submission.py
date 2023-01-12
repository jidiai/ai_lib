from pathlib import Path
import os
current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, "critic_10000.pth")
encoder_path = os.path.join(current_path, "encoder_10000.pth")

STATE_DIM = 128
ACTION_DIM = 7
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYER = 1

import torch
import torch.nn as nn
import torch.nn.functional as F

cnn_input_chanel = 3
cnn_channel_list = [32,32]
cnn_kernel_list = [4,2]
cnn_stride_list = [1,2]

class Flatten(nn.Module):
    def forward(self, x):
        """
        x: [batch_size, ...]
        """
        return x.reshape(x.size(0), -1)

class CNNEncoder(nn.Module):
    def __init__(self, input_chanel, hidden_size, output_size, channel_list, kernel_list, stride_list,
                 padding_list=None, batch_norm=False, pooling=False):
        super(CNNEncoder, self).__init__()
        assert len(channel_list) == len(kernel_list) == len(stride_list)
        if padding_list is None:
            padding_list = [0]*len(channel_list)

        net_list = []
        for idx in range(len(channel_list)):
            net_list.append(
                nn.Conv2d(in_channels=input_chanel, out_channels=channel_list[idx],
                          kernel_size=kernel_list[idx], stride=stride_list[idx],padding=padding_list[idx])
            )
            if batch_norm:
                net_list.append(nn.BatchNorm2d(channel_list[idx]))
            net_list.append(nn.ReLU())
            input_chanel = channel_list[idx]

        net_list.append(Flatten())
        # net_list.append(nn.Linear(hidden_size, output_size))
        # net_list.append(nn.ReLU())

        self.net = nn.Sequential(*net_list)

    def forward(self, state):
        # [batch, xx,xx,3]
        assert len(state.shape)==4
        state = state/255.0
        state = state.permute(0,3,1,2)
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        return x
#
cnn_encoder = CNNEncoder(input_chanel=cnn_input_chanel,
                         hidden_size=None,
                         output_size=None,
                         channel_list=cnn_channel_list,
                         kernel_list=cnn_kernel_list,
                         stride_list=cnn_stride_list)
cnn_encoder.load_state_dict(torch.load(encoder_path))

value_fn = Critic(input_size=STATE_DIM,
                  output_size=ACTION_DIM,
                  hidden_size=HIDDEN_SIZE,
                  num_hidden_layer=NUM_HIDDEN_LAYER)
value_fn.load_state_dict(torch.load(model_path))


def my_controller(observation, action_space, is_act_continuous=True):
    obs = torch.tensor(observation['obs']['image']).unsqueeze(0)
    encoded_obs = cnn_encoder(obs)
    action = torch.argmax(value_fn(encoded_obs))
    onehot_a = [0]*ACTION_DIM
    onehot_a[action.item()]=1
    return [onehot_a]



