import copy
from light_malib.utils.logger import Logger
from gym.spaces import Box,Discrete
from light_malib.envs.connect_four.env import DefaultFeatureEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

share_backbone=False
FeatureEncoder=DefaultFeatureEncoder
action_space=FeatureEncoder.action_space
observation_space=FeatureEncoder.observation_space

def init_weights(m):
    if isinstance(m,(nn.Linear,nn.Conv2d)): 
        nn.init.orthogonal_(m.weight,gain=1.0)
        
def to_tensor(x):
    if isinstance(x,np.ndarray):
        x=torch.from_numpy(x)
    return x
        
class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1),
        )

    def forward(self,x):
        features=self.body(x)
        return features

class SymmetricModel(nn.Module):
    def __init__(self,cls,model_config, observation_space, action_space_, custom_config, initialization):
        super().__init__()
        self.model_0=cls(model_config, observation_space, action_space_, custom_config, initialization)
        self.model_1=cls(model_config, observation_space, action_space_, custom_config, initialization)
        # legacy
        self.rnn_layer_num=self.model_0.rnn_layer_num
        self.rnn_state_size=self.model_0.rnn_state_size
    
    def forward(self,observations,rnn_states,rnn_masks):
        observations=to_tensor(observations)
        rnn_states=copy.deepcopy(to_tensor(rnn_states))
        
        assert len(observations.shape)==2
        # B,1
        idices=observations[:,0:1].long()
        observations=observations[:,1:]
        preds_0,_=self.model_0(observations,rnn_states,rnn_masks)
        preds_1,_=self.model_1(observations,rnn_states,rnn_masks)
        preds=torch.stack([preds_0,preds_1],dim=-1)
        idices=idices.unsqueeze(dim=1).repeat(1,preds.shape[1],1)
        preds=torch.gather(preds,dim=-1,index=idices)
        preds=preds.squeeze(-1)
        return preds,rnn_states

class _Actor(nn.Module):
    def __init__(self,model_config, observation_space, action_space_, custom_config, initialization):
        super().__init__()
        self.base=BaseNet()
        self.pool=nn.MaxPool2d(kernel_size=(6,1))
        self.head=nn.Linear(32,1)
        # legacy
        self.rnn_layer_num=1
        self.rnn_state_size=1

        self.apply(init_weights)

    def forward(self,observations,rnn_states,rnn_masks):
        observations=to_tensor(observations).float()
        rnn_states=to_tensor(rnn_states)
        observations=observations.view(-1,6,7,4)
        observations=observations.permute(0,3,1,2)
        features=self.base(observations)
        # B,32,1,7
        features=self.pool(features)
        features=features.squeeze(-2)
        # B,7,32
        features=features.permute(0,2,1)
        logits=self.head(features)
        logits=logits.squeeze(-1)
        return logits,rnn_states
        
class _Critic(nn.Module):
    def __init__(self,model_config, observation_space, action_space, custom_config, initialization):
        super().__init__()
        self.base=BaseNet()
        self.pool=nn.AvgPool2d(kernel_size=(6,7))
        self.head=nn.Linear(32,1)
        # legacy
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
        self.apply(init_weights)
        
    def forward(self,observations,rnn_states,rnn_masks):
        observations=to_tensor(observations).float()
        rnn_states=to_tensor(rnn_states)
        observations=observations.view(-1,6,7,4)
        observations=observations.permute(0,3,1,2)
        features=self.base(observations)
        # B,32,1,1
        features=self.pool(features)
        # B,32
        features=features.squeeze(-1).squeeze(-1)
        # B,1
        values=self.head(features)
        return values,rnn_states
    
class Actor(SymmetricModel):
    def __init__(self, model_config, observation_space, action_space_, custom_config, initialization):
        super().__init__(_Actor, model_config, observation_space, action_space_, custom_config, initialization)

class Critic(SymmetricModel):
    def __init__(self, model_config, observation_space, action_space_, custom_config, initialization):
        super().__init__(_Critic, model_config, observation_space, action_space_, custom_config, initialization)