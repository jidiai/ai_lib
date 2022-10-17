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

    def forward(self,**kwargs):
        explore_cfg=kwargs["explore_cfg"]
        mode=explore_cfg["mode"]
        q_values=kwargs[EpisodeKey.STATE_ACTION_VALUE]
        action_masks=kwargs[EpisodeKey.ACTION_MASK]
        q_values=to_tensor(q_values)
        action_masks=to_tensor(action_masks)
        # TODO(jh): a very small value?
        # print(q_values,action_masks)
        # assume masking invalid actions is done in critic
        # q_values=action_masks*q_values+(1-action_masks)*(-10e9)
        assert len(q_values.shape)==2,q_values.shape
        if mode=="greedy":
            actions=torch.argmax(q_values,dim=-1,keepdim=True)
            action_probs=torch.zeros_like(q_values)
            action_probs[torch.arange(action_probs.shape[0],device=action_probs.device),actions[:,0]]=1.0
        elif mode=="epsilon_greedy":
            epsilon=explore_cfg["epsilon"]
            best_actions=torch.argmax(q_values,dim=-1,keepdim=True)
            action_probs=action_masks/torch.sum(action_masks,dim=-1,keepdims=True)*epsilon
            action_probs[torch.arange(action_probs.shape[0],device=action_probs.device),best_actions[:,0]]+=(1-epsilon)
            # renormalize to avoid numerical issues
            action_probs=action_probs/torch.sum(action_probs,dim=-1,keepdims=True)
            actions=torch.multinomial(action_probs,num_samples=1)
            #print(q_values,action_probs,actions)
        elif mode=="softmax":
            temperature=explore_cfg["temperature"]
            q_values=q_values*temperature
            # mask again: ensure
            q_values=action_masks*q_values+(1-action_masks)*(-10e9)
            action_probs=torch.softmax(q_values,dim=-1)
            actions=torch.multinomial(action_probs,num_samples=1)
        else:
            raise NotImplementedError
        return actions,action_probs

