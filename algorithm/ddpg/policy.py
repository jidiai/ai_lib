import copy
import os
import pickle
import random
import gym
import torch
import numpy as np
from copy import deepcopy

from torch import nn
from utils.logger import Logger
from utils.typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from utils.episode import EpisodeKey

from gym.spaces import Box
from algorithm.common.policy import Policy
from ..utils import init_fc_weights
import wrapt
import tree
import importlib
from utils.logger import Logger
from registry import registry

from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from torch.autograd import Variable

from ..common.misc import onehot_from_logits, gumbel_softmax


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
class DDPG(nn.Module):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,  # legacy
        action_space: gym.spaces.Space,  # legacy
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs
    ):
        del observation_space
        del action_space

        self.registered_name = registered_name
        assert self.registered_name == "DDPG"
        self.model_config = model_config
        self.custom_config = custom_config

        super().__init__()

        model_type = model_config["model"]
        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("model.{}".format(model_type))

        self.encoder = model.Encoder()
        self._rewarder = model.Rewarder()

        self.observation_space = self.encoder.observation_space
        self.action_space = self.encoder.action_space
        assert isinstance(self.action_space, Box), str(self.action_space)

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

        self.target_critic = deepcopy(self.critic)
        self.target_actor = deepcopy(self.actor)
        self.discrete_action = self.custom_config["discrete_action"]

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
    def feature_encoder(self):  # legacy
        return self.encoder

    @property
    def rewarder(self):
        return self._rewarder

    def get_initial_state(self, batch_size):
        if hasattr(self.critic, "get_initial_state"):
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
        step = kwargs.get("step", 0)
        to_numpy = kwargs.get("to_numpy", True)
        explore = kwargs["explore"]
        with torch.no_grad():
            obs = kwargs[EpisodeKey.CUR_OBS]
            action_masks = kwargs[EpisodeKey.ACTION_MASK]
            if self.discrete_action:
                pi = self.actor(
                    **{EpisodeKey.CUR_OBS: obs, EpisodeKey.ACTION_MASK: action_masks}
                )
                if explore:
                    pi = gumbel_softmax(pi, temperature=1.0, hard=True)
                else:
                    pi = onehot_from_logits(pi)
            else:
                pi = self.actor(
                    **{EpisodeKey.CUR_OBS: obs, EpisodeKey.ACTION_MASK: action_masks}
                )
                # if explore:
                #     logits += torch.autograd.Variable(
                #         torch.Tensor(np.random.standard_normal(logits.shape)),
                #         requires_grad=False,
                #     )
                pi = pi.detach().cpu().numpy()

        return {EpisodeKey.ACTION: pi}

    def compute_actions_by_target_actor(self, obs):
        with torch.no_grad():
            pi = self.target_actor(obs)
            if self.discrete_action:
                pi = onehot_from_logits(pi)
        return pi

    @shape_adjusting
    def value_function(self, **kwargs):
        to_numpy = kwargs.get("to_numpy", True)
        use_target_critic = kwargs.get("use_target_critic", False)
        if use_target_critic:
            critic = self.critic
        else:
            critic = self.target_critic
        with torch.no_grad():
            # obs = kwargs[EpisodeKey.CUR_OBS]
            # action_masks = kwargs[EpisodeKey.ACTION_MASK]
            obs_action = kwargs[EpisodeKey.OBS_ACTION]
            q_values = critic(**{EpisodeKey.OBS_ACTION: obs_action})
            # denormalize
            # if hasattr(self,"value_normalizer"):
            #     q_values=self.value_normalizer.denormalize(q_values)
            if to_numpy:
                q_values = q_values.cpu().numpy()
        return {
            EpisodeKey.STATE_ACTION_VALUE: q_values,
        }

    def dump(self, dump_dir):
        torch.save(
            self.critic.state_dict(), os.path.join(dump_dir, "critic_state_dict.pt")
        )
        torch.save(
            self.actor.state_dict(), os.path.join(dump_dir, "actor_state_dict.pt")
        )
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        policy = DDPG(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )

        critic_path = os.path.join(dump_dir, "critic_state_dict.pt")
        if os.path.exists(critic_path):
            critic_state_dict = torch.load(
                os.path.join(dump_dir, "critic_state_dict.pt"), policy.device
            )
            policy.critic.load_state_dict(critic_state_dict)
            policy.target_critic = deepcopy(policy.critic)

        actor_path = os.path.join(dump_dir, "actor_state_dict.pt")
        if os.path.exists(actor_path):
            actor_state_dict = torch.load(
                os.path.join(dump_dir, "actor_state_dict.pt"), policy.device
            )
            policy.actor.load_state_dict(actor_state_dict)
            policy.target_actor = deepcopy(policy.actor)

        return policy

    # def compute_actions_by_target_actor(
    #     self, observation: DataTransferType, **kwargs
    # ) -> DataTransferType:
    #     with torch.no_grad():
    #         pi = self.target_actor(observation)
    #         if self._discrete_action:
    #             pi = misc.onehot_from_logits(pi)
    #     return pi

    # def update_target(self):
    #     self.target_critic.load_state_dict(self._critic.state_dict())
    #     self.target_actor.load_state_dict(self._actor.state_dict())
    #
    # def soft_update(self, tau=0.01):
    #     misc.soft_update(self.target_critic, self.critic, tau)
    #     misc.soft_update(self.target_actor, self.actor, tau)
