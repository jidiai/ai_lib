import copy
import os
import pickle
import random
import gym
import torch
import numpy as np

from torch import nn
from utils.logger import Logger
from utils._typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from utils.episode import EpisodeKey
from gym.spaces import Discrete
from algorithm.common.policy import Policy
from copy import deepcopy

from ..utils import PopArt, init_fc_weights
import wrapt
import tree
import importlib
from utils.logger import Logger
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
class PPO(nn.Module):
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
        assert self.registered_name=='PPO'
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
        self.state_space = self.encoder.state_space
        assert isinstance(self.action_space, Discrete), str(self.action_space)

        self.actor = model.Actor(
            self.model_config["actor"],
            self.observation_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        self.critic = model.Critic(
            self.model_config["critic"],
            self.state_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )
        # breakpoint()

        self.target_critic = deepcopy(self.critic)

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


    def get_initial_state(self, batch_size) -> List[DataTransferType]:
        if hasattr(self.critic, "get_initial_state"):
            return {
                EpisodeKey.ACTOR_RNN_STATE: self.actor.get_initial_state(batch_size),
                EpisodeKey.CRITIC_RNN_STATE: self.critic.get_initial_state(batch_size)
            }
        else:
            return {}

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.to(device)
        self_copy.device = device
        return self_copy


    def compute_actions(self, observation, **kwargs):
        raise RuntimeError("Shouldn't use it currently")

    def forward_actor(self, obs, actor_rnn_states, rnn_masks):
        logits, actor_rnn_states = self.actor(obs, actor_rnn_states, rnn_masks)
        return logits, actor_rnn_states

    @shape_adjusting
    def compute_action(self, **kwargs):
        with torch.no_grad():
            observations = kwargs[EpisodeKey.CUR_OBS]
            action_masks = kwargs[EpisodeKey.ACTION_MASK]
            rnn_masks = kwargs[EpisodeKey.DONE]

            # actions, action_probs = self.actor(
            #     **{EpisodeKey.CUR_OBS: observations, EpisodeKey.ACTION_MASK:action_masks}
            # )
            logits = self.actor(
                **{EpisodeKey.CUR_OBS: observations}
            )
            logits-=1e10*(1-action_masks)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().unsqueeze(-1).cpu().numpy()
            action_probs = dist.probs.detach().cpu().numpy()
            # if self.random_exploration:
            #     exploration_actions = np.zeros(actions.shape, dtype=int)
            #     for i in range(len(actions)):
            #         if random.uniform(0, 1) < self.random_exploration:
            #             exploration_actions[i] = int(random.choice(range(19)))
            #         else:
            #             exploration_actions[i] = int(actions[i])
            #     actions = exploration_actions


            return {
                EpisodeKey.ACTION: actions,
                EpisodeKey.ACTION_DIST: action_probs,
            }

    @shape_adjusting
    def value_function(self, **kwargs):
        with torch.no_grad():
            # FIXME(ziyu): adjust shapes
            if EpisodeKey.CUR_STATE not in kwargs:
                states = kwargs[EpisodeKey.CUR_OBS]
            else:
                states = kwargs[EpisodeKey.CUR_STATE]
            critic_rnn_state = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            rnn_mask = kwargs[EpisodeKey.DONE]
            value, _ = self.critic(states, critic_rnn_state, rnn_mask)
            value = value.cpu().numpy()
            return {EpisodeKey.STATE_VALUE: value}

    def train(self):
        pass

    def eval(self):
        pass

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def dump(self, dump_dir):
        torch.save(self.actor, os.path.join(dump_dir, "actor.pt"))
        torch.save(self.critic, os.path.join(dump_dir, "critic.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        res = PPO(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )

        actor_path = os.path.join(dump_dir, "actor.pt")
        critic_path = os.path.join(dump_dir, "critic.pt")
        if os.path.exists(actor_path):
            actor = torch.load(os.path.join(dump_dir, "actor.pt"), res.device)
            hard_update(res.actor, actor)
        if os.path.exists(critic_path):
            critic = torch.load(os.path.join(dump_dir, "critic.pt"), res.device)
            hard_update(res.critic, critic)
        return res

    # XXX(ziyu): test for this policy
    def state_dict(self):
        """Return state dict in real time"""

        res = {
            k: copy.deepcopy(v).cpu().state_dict()
            if isinstance(v, nn.Module)
            else v.state_dict()
            for k, v in self._state_handler_dict.items()
        }
        return res


# if __name__ == "__main__":
#     from light_malib.envs.gr_football import env, default_config
#     import yaml

#     cfg = yaml.load(open("mappo_grfootball/mappo_5_vs_5.yaml"))
#     env = env(**default_config)
#     custom_cfg = cfg["algorithms"]["MAPPO"]["custom_config"]
#     custom_cfg.update({"global_state_space": env.observation_spaces})
#     policy = MAPPO(
#         "MAPPO",
#         env.observation_spaces["team_0"],
#         env.action_spaces["team_0"],
#         cfg["algorithms"]["MAPPO"]["model_config"],
#         custom_cfg,
#         env_agent_id="team_0",
#     )
#     os.makedirs("play")
#     policy.dump("play")
#     MAPPO.load("play", env_agent_id="team_0")
