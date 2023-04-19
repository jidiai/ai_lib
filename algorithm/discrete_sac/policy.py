import copy
import os
import pickle
import random
import gym
import torch
import numpy as np
from gym.spaces import Discrete

from torch.distributions import Categorical, Normal
from torch import nn
from utils.logger import Logger
from utils._typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from utils.episode import EpisodeKey

from algorithm.common.policy import Policy

# from malib.utils.typing import DataTransferType, Dict, Tuple, BehaviorMode
# from malib.algorithm.common.model import get_model
# from malib.algorithm.common import misc

from ..utils import PopArt, init_fc_weights
import wrapt
import tree
import importlib
from utils.logger import Logger
from registry import registry
from copy import deepcopy


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
class DiscreteSAC(nn.Module):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):

        self.registered_name = registered_name
        assert self.registered_name == "DiscreteSAC"
        self.model_config = model_config
        self.custom_config = custom_config

        super().__init__()

        model_type = model_config["model"]
        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("model.{}".format(model_type))

        self.encoder = model.Encoder()
        if hasattr(model, "Rewarder"):
            self._rewarder = model.Rewarder()

        self.observation_space = self.encoder.observation_space
        self.action_space = self.encoder.action_space
        self.state_space = self.encoder.state_space

        assert isinstance(self.action_space, Discrete), str(self.action_space)

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
        self.target_actor = deepcopy(self.actor)

        self.critic_1 = model.Critic(
            self.model_config["critic"],
            self.state_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )
        self.target_critic_1 = deepcopy(self.critic_1)

        self.critic_2 = model.Critic(
            self.model_config["critic"],
            self.state_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )
        self.target_critic_2 = deepcopy(self.critic_2)

        if self.custom_config.get("use_auto_alpha", False):
            self.use_auto_alpha = True
            self._target_entropy = 0.98 * np.log(np.prod(action_space.n))
            self._log_alpha = torch.zeros(1, requires_grad=True)
            self._alpha = self._log_alpha.detach().exp()
        else:
            self.use_auto_alpha = False
            self._alpha = self.custom_config.get("alpha", 0.05)

        self.current_eps = 0

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

    # @property
    # def rewarder(self):
    #     return self._rewarder

    def get_initial_state(self, batch_size):
        if hasattr(self.critic_1, "get_initial_state"):
            return {
                EpisodeKey.CRITIC_RNN_STATE: self.critic_1.get_initial_state(batch_size)
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
            logits = self.actor(**{EpisodeKey.CUR_OBS: obs})
            assert len(logits.shape) > 1, logits.shape

            logits = logits * action_masks
            m = Categorical(logits=logits)
            action_probs = m.probs.detach().cpu().numpy()
            actions = m.sample().detach().unsqueeze(-1).cpu().numpy()

            return {EpisodeKey.ACTION: actions, EpisodeKey.ACTION_PROBS: action_probs}

    @shape_adjusting
    def value_function(self, **kwargs):
        to_numpy = kwargs.get("to_numpy", True)
        use_target_critic = kwargs.get("use_target_critic", False)
        if use_target_critic:
            critic_1 = self.critic_1
            critic_2 = self.critic_2
        else:
            critic_1 = self.target_critic_1
            critic_2 = self.target_critic_2

        with torch.no_grad():
            obs = kwargs[EpisodeKey.CUR_STATE]
            action_masks = kwargs[EpisodeKey.ACTION_MASK]
            q_values_1 = critic_1(
                **{EpisodeKey.CUR_OBS: obs, EpisodeKey.ACTION_MASK: action_masks}
            )
            q_values_2 = critic_2(
                **{EpisodeKey.CUR_OBS: obs, EpisodeKey.ACTION_MASK: action_masks}
            )
            # denormalize
            # if hasattr(self,"value_normalizer"):
            #     q_values=self.value_normalizer.denormalize(q_values)
            if to_numpy:
                q_values_1 = q_values_1.cpu().numpy()
                q_values_2 = q_values_2.cpu().numpy()

        return {
            EpisodeKey.STATE_ACTION_VALUE: [q_values_1, q_values_2],
            EpisodeKey.ACTION_MASK: action_masks,
        }

    def dump(self, dump_dir):
        torch.save(
            self.actor.state_dict(), os.path.join(dump_dir, "actor_state_dict.pt")
        )
        torch.save(
            self.critic_1.state_dict(), os.path.join(dump_dir, "critic1_state_dict.pt")
        )
        torch.save(
            self.critic_2.state_dict(), os.path.join(dump_dir, "critic2_state_dict.pt")
        )
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        policy = DiscreteSAC(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )

        critic1_path = os.path.join(dump_dir, "critic1_state_dict.pt")
        if os.path.exists(critic1_path):
            critic1_state_dict = torch.load(
                os.path.join(dump_dir, "critic1_state_dict.pt"), policy.device
            )
            policy.critic_1.load_state_dict(critic1_state_dict)
            policy.target_critic_1 = deepcopy(policy.critic_1)

        critic2_path = os.path.join(dump_dir, "critic2_state_dict.pt")
        if os.path.exists(critic2_path):
            critic2_state_dict = torch.load(
                os.path.join(dump_dir, "critic2_state_dict.pt"), policy.device
            )
            policy.critic_2.load_state_dict(critic2_state_dict)
            policy.target_critic_2 = deepcopy(policy.critic_2)

        return policy
