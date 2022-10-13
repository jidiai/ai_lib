import copy
import os
import pickle
import random
import gym
import torch
import numpy as np

from torch import nn
from utils.logger import Logger
from utils.typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from utils.episode import EpisodeKey

import wrapt
import tree
import importlib
from utils.logger import Logger
from gym.spaces import Discrete
from ..utils import PopArt
from registry import registry

def hard_update(target, source):
    """Copy network parameters from source to target.

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15

    :param torch.nn.Module target: Net to copy parameters to.
    :param torch.nn.Module source: Net whose parameters to copy
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

@wrapt.decorator
def shape_adjusting(wrapped, instance, args, kwargs):
    """
    A wrapper that adjust the inputs to corrent shape.
    e.g.
        given inputs with shape (n_rollout_threads, n_agent, ...)
        reshape it to (n_rollout_threads * n_agent, ...)
    """
    offset = len(instance.observation_space.shape)
    original_shape_pre = kwargs[EpisodeKey.CUR_OBS].shape[:-offset]
    num_shape_ahead = len(original_shape_pre)

    def adjust_fn(x):
        if isinstance(x, np.ndarray):
            return np.reshape(x, (-1,) + x.shape[num_shape_ahead:])
        else:
            return x

    def recover_fn(x):
        if isinstance(x, np.ndarray):
            return np.reshape(x, original_shape_pre + x.shape[1:])
        else:
            return x

    adjusted_args = tree.map_structure(adjust_fn, args)
    adjusted_kwargs = tree.map_structure(adjust_fn, kwargs)

    rets = wrapped(*adjusted_args, **adjusted_kwargs)

    recover_rets = tree.map_structure(recover_fn, rets)

    return recover_rets

@registry.registered(registry.POLICY)
class QLearning(nn.Module):
    def __init__(
        self,
        registered_name: str,                   
        observation_space: gym.spaces.Space,    # legacy
        action_space: gym.spaces.Space,         # legacy
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        del observation_space
        del action_space
        
        self.registered_name=registered_name
        assert self.registered_name=="QLearning"
        self.model_config=model_config
        self.custom_config=custom_config
        
        super().__init__()
       
        model_type = model_config["model"]
        Logger.warning("use model type: {}".format(model_type))
        model=importlib.import_module("model.{}".format(model_type))
        
        self.encoder=model.Encoder()
        
        # TODO(jh): extension to multi-agent cooperative case
        # self.env_agent_id = kwargs["env_agent_id"]
        # self.global_observation_space=self.encoder.global_observation_space if hasattr(self.encoder,"global_observation_space") else self.encoder.observation_space
        self.observation_space=self.encoder.observation_space
        self.action_space=self.encoder.action_space
        assert isinstance(self.action_space,Discrete),str(self.action_space)

        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )
        
        self.actor = model.Actor(
            self.model_config["actor"],
            self.observation_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],            
        )
    
        self.critic = model.Critic(
            self.model_config["critic"],
            self.observation_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        # if custom_config["use_popart"]:
        #     self.value_normalizer = PopArt(
        #         1, device=self.device, beta=custom_config["popart_beta"]
        #     )
            
    @property
    def description(self):
        """Return a dict of basic attributes to identify policy.

        The essential elements of returned description:

        - registered_name: `self.registered_name`
        - observation_space: `self.observation_space`
        - action_space: `self.action_space`
        - model_config: `self.model_config`
        - custom_config: `self.custom_config`

        :return: A dictionary.
        """

        return {
            "registered_name": self.registered_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "model_config": self.model_config,
            "custom_config": self.custom_config,
        }
        
    @property
    def feature_encoder(self): # legacy
        return self.encoder

    def get_initial_state(self, batch_size):
        if hasattr(self.critic,"get_initial_state"):
            return {
                EpisodeKey.CRITIC_RNN_STATE: self.critic.get_initial_state(batch_size)
            }
        else:
            return {}

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.to(device)
        self_copy.device = device
        return self_copy

    @shape_adjusting
    def compute_action(self, **kwargs):   
        '''
        TODO(jh): need action sampler, e.g. epsilon-greedy.
        '''
        step=kwargs.get("step",0)
        to_numpy=kwargs.get("to_numpy",True)
        explore=kwargs["explore"]
        with torch.no_grad():
            obs=kwargs[EpisodeKey.CUR_OBS]
            action_masks=kwargs[EpisodeKey.ACTION_MASK]
            q_values=self.critic(**{EpisodeKey.CUR_OBS:obs,EpisodeKey.ACTION_MASK:action_masks})
            # denormalize
            # if hasattr(self,"value_normalizer"):
            #     q_values=self.value_normalizer.denormalize(q_values)
            if not explore:
                explore_cfg={"mode":"greedy"} 
            else:
                _explore_cfg=self.custom_config["explore_cfg"]
                assert _explore_cfg["mode"]=="epsilon_greedy","only epsilon_greedy is supported now."
                if "epsilon" not in _explore_cfg:
                    # only support linear decaying now
                    max_epsilon=_explore_cfg["max_epsilon"]
                    min_epsilon=_explore_cfg["min_epsilon"]
                    total_decay_steps=_explore_cfg["total_decay_steps"]
                    epsilon=(max_epsilon-min_epsilon)/total_decay_steps*(total_decay_steps-step+1)+min_epsilon
                    explore_cfg=copy.deepcopy(_explore_cfg)
                    explore_cfg={"mode":"epsilon_greedy","epsilon":epsilon}
                else:
                    assert "max_epsilon" not in _explore_cfg and "min_epsilon" not in _explore_cfg and "total_decay_steps" not in _explore_cfg
                    explore_cfg=copy.deepcopy(_explore_cfg)
            actions,action_probs=self.actor(**{EpisodeKey.STATE_ACTION_VALUE:q_values,EpisodeKey.ACTION_MASK:action_masks},explore_cfg=explore_cfg)
            if to_numpy:
                actions=actions.cpu().numpy()
                action_probs=action_probs.cpu().numpy()
        return {EpisodeKey.ACTION:actions,EpisodeKey.ACTION_PROBS:action_probs}
            
    @shape_adjusting
    def value_function(self, **kwargs):
        to_numpy=kwargs.get("to_numpy",True)
        with torch.no_grad():
            obs=kwargs[EpisodeKey.CUR_OBS]
            action_masks=kwargs[EpisodeKey.ACTION_MASK]
            q_values=self.critic(**{EpisodeKey.CUR_OBS:obs,EpisodeKey.ACTION_MASK:action_masks})
            # denormalize
            # if hasattr(self,"value_normalizer"):
            #     q_values=self.value_normalizer.denormalize(q_values)
            if to_numpy:
                q_values=q_values.cpu().numpy()
        return {EpisodeKey.STATE_ACTION_VALUE: q_values,
                EpisodeKey.ACTION_MASK: action_masks}

    def dump(self, dump_dir):
        torch.save(self.critic.state_dict(), os.path.join(dump_dir, "critic_state_dict.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        policy = QLearning(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )
        
        critic_path=os.path.join(dump_dir,"critic_state_dict.pt")
        if os.path.exists(critic_path):
            critic_state_dict = torch.load(os.path.join(dump_dir, "critic_state_dict.pt"), policy.device)
            policy.critic.load_state_dict(critic_state_dict)
        return policy