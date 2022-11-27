# -*- coding: utf-8 -*-
from tkinter import TRUE
import torch
import torch.nn.functional as F
from utils.episode import EpisodeKey
from algorithm.common.loss_func import LossFunc
from utils.logger import Logger
from registry import registry


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


def soft_update(target, source, tau):
    """Perform DDPG soft update (move target params toward source based on weight factor tau).

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11

    :param torch.nn.Module target: Net to copy parameters to
    :param torch.nn.Module source: Net whose parameters to copy
    :param float tau: Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


@registry.registered(registry.LOSS)
class DeepQLearningLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super().__init__()

        self._use_huber_loss = False
        if self._use_huber_loss:
            self.huber_delta = 10.0

        self._use_max_grad_norm = True

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()
        self.step_ctr = 0

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""

        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "critic": optim_cls(
                    self.policy.critic.parameters(), lr=self._params["critic_lr"]
                ),
            }
        else:
            self.optimizers["critic"].param_groups = []
            self.optimizers["critic"].add_param_group(
                {"params": self.policy.critic.parameters()}
            )

    def loss_compute(self, sample):
        self.step_ctr += 1
        policy = self._policy
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)
        self.gamma = policy.custom_config["gamma"]
        target_update_freq = policy.custom_config["target_update_freq"]
        target_update_lr = policy.custom_config["target_update_lr"]

        (
            observations,
            action_masks,
            actions,
            rewards,
            dones,
            next_observations,
            next_action_masks,
        ) = (
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.REWARD],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.NEXT_OBS],
            sample[EpisodeKey.NEXT_ACTION_MASK],
        )

        with torch.no_grad():
            next_q_values = policy.value_function(
                **{
                    EpisodeKey.CUR_OBS: next_observations,
                    EpisodeKey.ACTION_MASK: next_action_masks,
                },
                to_numpy=False,
                use_target_critic=TRUE
            )[EpisodeKey.STATE_ACTION_VALUE]
            # denormalize
            # if hasattr(self,"value_normalizer"):
            #     next_q_values=self.value_normalizer.denormalize(next_q_values)
        # breakpoint()
        # rewards = (rewards-rewards.mean())/(rewards.std()+1e-10)

        targets = (
            rewards
            + (1 - dones)
            * self.gamma
            * torch.max(next_q_values, dim=-1, keepdim=True)[0]
        )

        # TODO(jh): how to use popart?
        # TODO value normalizer shoule be put inside critic!
        # if policy.custom_config["use_popart"]:
        #     targets=policy.value_normalizer(targets)

        q_values = policy.critic(
            **{EpisodeKey.CUR_OBS: observations, EpisodeKey.ACTION_MASK: action_masks}
        )
        selected_q_values = torch.gather(q_values, dim=-1, index=actions)

        if self._use_huber_loss:
            value_loss = huber_loss(
                targets - selected_q_values, self.huber_delta
            ).mean()
        else:
            value_loss = mse_loss(targets - selected_q_values).mean()

        self.optimizers["critic"].zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.critic.parameters(), self.max_grad_norm
            )
        self.optimizers["critic"].step()

        # update target softly
        if self.step_ctr % target_update_freq == 0:
            soft_update(
                self._policy.target_critic, self._policy.critic, tau=target_update_lr
            )

        stats = {"value_loss": float(value_loss.detach().cpu().numpy())}
        return stats

    def zero_grad(self):
        pass

    def step(self):
        pass
