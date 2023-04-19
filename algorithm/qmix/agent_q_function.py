import torch
import torch.nn as nn
import numpy as np
import copy

def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input

### https://github.com/marlbenchmark/off-policy/blob/release/offpolicy/algorithms/qmix/algorithm/agent_q_function.py
class AgentQFunction(nn.Module):
    """
    Individual agent q network (RNN).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, act_dim, device=None):
        super(AgentQFunction, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._use_rnn_layer = args.use_rnn_layer
        self._gain = args.gain
        # self.device = device
        # self.tpdv = dict(dtype=torch.float32, device=device)

        if self._use_rnn_layer:
            self.rnn = RNNBase(args, input_dim)
        else:
            self.mlp = MLPBase(args, input_dim)

        self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain)

        # self.to(device)

    def forward(self, obs, rnn_states):
        """
        Compute q values for every action given observations and rnn states.
        :param obs: (torch.Tensor) observations from which to compute q values.
        :param rnn_states: (torch.Tensor) rnn states with which to compute q values.
        :return q_outs: (torch.Tensor) q values for every action
        :return h_final: (torch.Tensor) new rnn states
        """
        obs = to_torch(obs)     #.to(**self.tpdv)
        rnn_states = to_torch(rnn_states)       #.to(**self.tpdv)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            # obs is now of shape (seq_len, batch_size, obs_dim)
        if len(rnn_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_states = rnn_states[None]

        inp = obs

        if self._use_rnn_layer:
            rnn_outs, h_final = self.rnn(inp, rnn_states)
        else:
            rnn_outs = self.mlp(inp)
            h_final = rnn_states[0, :, :]

        # pass outputs through linear layer
        q_outs = self.q(rnn_outs, no_sequence)

        return q_outs, h_final


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)
        del self.fc_h

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class CONVLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, use_orthogonal, use_ReLU):
        super(CONVLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.conv = nn.Sequential(
                init_(nn.Conv1d(in_channels=input_dim, out_channels=hidden_size//4, kernel_size=3, stride=2, padding=0)), active_func, #nn.BatchNorm1d(hidden_size//4),
                init_(nn.Conv1d(in_channels=hidden_size//4, out_channels=hidden_size//2, kernel_size=3, stride=1, padding=1)), active_func, #nn.BatchNorm1d(hidden_size//2),
                init_(nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)), active_func)#, nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        x = self.conv(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, inputs_dim):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._use_conv1d = args.use_conv1d
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(inputs_dim)

        if self._use_conv1d:
            self.conv = CONVLayer(self._stacked_frames, self.hidden_size, self._use_orthogonal, self._use_ReLU)
            random_x = torch.FloatTensor(1, self._stacked_frames, inputs_dim)
            random_out = self.conv(random_x)
            assert len(random_out.shape) == 3
            inputs_dim = random_out.size(-1) * random_out.size(-2)

        self.mlp = MLPLayer(inputs_dim, self.hidden_size,
                            self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        return x


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs):
        self.rnn.flatten_parameters()
        x, hxs = self.rnn(x, hxs)
        x = self.norm(x)
        return x, hxs[0, :, :]

class RNNBase(MLPBase):
    def __init__(self, args, inputs_dim):
        super(RNNBase, self).__init__(args, inputs_dim)

        self._recurrent_N = args.recurrent_N

        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

    def forward(self, x, hxs):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        x, hxs = self.rnn(x,hxs)

        return x, hxs


class ACTLayer(nn.Module):
    def __init__(self, act_dim, hidden_size, use_orthogonal, gain):
        super(ACTLayer, self).__init__()

        self.multi_discrete = False
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if isinstance(act_dim, np.ndarray):
            # MultiDiscrete setting: have n Linear layers for each action
            self.multi_discrete = True
            self.action_outs = nn.ModuleList([init_(nn.Linear(hidden_size, a_dim)) for a_dim in act_dim])
        else:
            self.action_out = init_(nn.Linear(hidden_size, act_dim))

    def forward(self, x, no_sequence=False):

        if self.multi_discrete:
            act_outs = []
            for a_out in self.action_outs:
                act_out = a_out(x)
                if no_sequence:
                    # remove the dummy first time dimension if the input didn't have a time dimension
                    act_out = act_out[0, :, :]
                act_outs.append(act_out)
        else:
            act_outs = self.action_out(x)
            if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                act_outs = act_outs[0, :, :]

        return act_outs