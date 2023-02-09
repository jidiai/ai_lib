from utils.episode import EpisodeKey

import torch
import torch.nn as nn
import numpy as np

def to_tensor(arr):
    if isinstance(arr,np.ndarray):
        arr=torch.FloatTensor(arr)
    return arr

class Actor(nn.Module):
    def __init__(
            self,
            model_config,
            observation_space,
            action_space,
            custom_config,
            initialization,
    ):
        super().__init__()

        in_dim = observation_space.shape[0]
        out_dim = action_space.n
        hidden_size=128

        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        self._init()

    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.uniform_(module.bias, 0, 0.001)
            else:
                pass


    def forward(self,**kwargs):
        obs = kwargs[EpisodeKey.CUR_OBS]
        # action_masks=kwargs[EpisodeKey.ACTION_MASK]

        obs = to_tensor(obs)
        # action_masks=to_tensor(action_masks)
        # TODO(jh): a very small value?

        logits = self.actor(obs)
        return logits
        #
        # logits -= 1e10*(1-action_masks)
        # dist = torch.distributions.Categorical(logits = logits)
        # actions = dist.sample()
        # action_prob = dist.log_prob(actions).exp()

        # _action_prob = torch.softmax(self.actor(obs), dim=-1)
        # action_prob_masked = _action_prob*action_masks
        # action_prob = action_prob_masked/sum(action_prob_masked)
        # actions = torch.multinomial(action_prob, num_samples=1)

        # return actions.unsqueeze(-1), dist.probs

    # def logits(self, **kwargs):
    #     obs = kwargs[EpisodeKey.CUR_OBS]
    #     obs = to_tensor(obs)
    #     return self.actor(obs)