import torch.nn as nn
import torch.nn.functional as F


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


class Dueling_Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear1(x))

        # value
        y1 = self.linear2(x1)
        # advantage
        y2 = self.linear3(x2)
        # y2.mean(dim=1,keepdim=True)：dim=1按行取均值；keepdim=True时，输出与输入维度相同，仅仅时输出在求均值的维度上元素个数变为1。
        # keepdim=False时，输出比输入少一个维度，就是指定的dim求均值的维度。
        x3 = y1 + y2 - y2.mean(dim=1, keepdim=True)

        return x3

      
class openai_critic(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, 128)
        self.linear_c2 = nn.Linear(128, 64)
        self.linear_c = nn.Linear(64, 1)
        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


########################### MAgent #################################
import torch

def weights_init_kaiming(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class CNN_encoder(nn.Module):
    """
    view size : [batch, view_space]
    """

    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()
        )

    def forward(self, view_state):
        """
        [batch, 10,10,5]
        """
        state = view_state.permute(0, 3, 1, 2)

        return self.net(state)  # [batch, 128]


class CNN_Critic(nn.Module):
    def __init__(self, output_size, input_size=128, hidden_size=128, if_feature=False, feature_size=None):
        super().__init__()

        self.conv = CNN_encoder()

        self.if_feature = if_feature

        self.input_size = input_size
        self.output_size = output_size

        if if_feature:
            self.view_linear = nn.Linear(input_size, hidden_size)
            self.feature_linear = nn.Linear(feature_size, feature_size)
            self.concat_linear = nn.Sequential(
                nn.Linear(hidden_size + feature_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            self.view_linear = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        self.apply(weights_init_kaiming)

    def forward(self, view_state, feature_state=None):
        """
        view_state: [batch, 10,10,5]
        feature_state: [batch, 34] or None
        """
        if self.if_feature:
            view_encoded_state = self.view_linear(self.conv(view_state))
            view_encoded_state = F.relu(view_encoded_state)

            feature_encoded_state = F.relu(self.feature_linear(feature_state))
            concate = torch.cat([view_encoded_state, feature_encoded_state], dim=-1)  # []batch, hidden+feature
            return self.concat_linear(concate)  # [batch, output_size]

        else:
            conv_encoded = self.conv(view_state)  # [batch, 128]
            # print('conv encoded shape', conv_encoded.shape)
            return self.view_linear(conv_encoded)
