# -*- coding: utf-8 -*-
import torch
from utils.episode import EpisodeKey
from algorithm.common.loss_func import LossFunc
from utils.logger import Logger
import torch.nn.functional as F


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return (e ** 2) / 2


def to_value(tensor: torch.Tensor):
    return tensor.detach().cpu().item()


def basic_stats(name, tensor: torch.Tensor):
    stats = {}
    stats["{}_max".format(name)] = to_value(tensor.max())
    stats["{}_min".format(name)] = to_value(tensor.min())
    stats["{}_mean".format(name)] = to_value(tensor.mean())
    stats["{}_std".format(name)] = to_value(tensor.std())
    return stats


class BCLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(BCLoss, self).__init__()
        self._use_max_grad_norm = True

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""

        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "actor": optim_cls(
                    self.policy.actor.parameters(),
                    lr=self._params["actor_lr"],
                    eps=self._params["opti_eps"],
                    weight_decay=self._params["weight_decay"],
                ),
                "critic": optim_cls(
                    self.policy.critic.parameters(), lr=self._params["critic_lr"]
                ),
            }
        else:
            self.optimizers["actor"].param_groups = []
            self.optimizers["actor"].add_param_group(
                {"params": self.policy.actor.parameters()}
            )
            self.optimizers["critic"].param_groups = []
            self.optimizers["critic"].add_param_group(
                {"params": self.policy.critic.parameters()}
            )

    def loss_compute(self, sample):
        policy = self._policy
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

        self._policy.opt_cnt += 1

        (
            obs_batch,
            actions_batch,
        ) = (sample[EpisodeKey.EXPERT_OBS], sample[EpisodeKey.EXPERT_ACTION].long())

        action_logits_batch, _ = self._policy.actor(obs_batch, None, None)

        policy_loss = F.cross_entropy(
            action_logits_batch.reshape(-1, action_logits_batch.shape[-1]),
            actions_batch.reshape(-1),
        )

        self.optimizers["actor"].zero_grad()
        policy_loss.backward()
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.actor.parameters(), self.max_grad_norm
            )
        self.optimizers["actor"].step()

        stats = dict(policy_loss=float(policy_loss.detach().cpu().numpy()))

        return stats

    def zero_grad(self):
        return super().zero_grad()

    def step(self):
        return super().step()
