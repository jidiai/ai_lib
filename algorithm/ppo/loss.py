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
class PPOLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super().__init__()

        self._use_clipped_value_loss = True
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
        self.step_ctr += 1
        policy = self._policy
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)


        (
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_probs_batch,
            available_actions_batch,
            dones_batch,
            adv_targ,
        ) = (
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.RETURN],
            sample.get(EpisodeKey.ACTIVE_MASK, None),
            sample[EpisodeKey.ACTION_DIST],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.ADVANTAGE],
        )
        # V = self.policy.critic(**{EpisodeKey.CUR_OBS: obs_batch, EpisodeKey.ACTIVE_MASK: available_actions_batch})
        # delta =
        if EpisodeKey.CUR_STATE in sample:
            share_obs_batch = sample[EpisodeKey.CUR_STATE]
        else:
            share_obs_batch = sample[EpisodeKey.CUR_OBS]

        values, action_log_probs, dist_entropy = self._evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            dones_batch,
            active_masks_batch,
        )
        actions_batch = actions_batch.reshape(actions_batch.shape[0], 1)
        old_action_log_probs_batch = torch.log(
            old_action_probs_batch.gather(-1, actions_batch)
        )
        imp_weights = torch.exp(
            action_log_probs.unsqueeze(-1) - old_action_log_probs_batch
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )
        assert active_masks_batch is None, print('only support None active mask')
        # if active_masks_batch is not None:
        #     surr = torch.min(surr1, surr2)
        #     policy_action_loss = (
        #         -torch.sum(surr, dim=-1, keepdim=True) * active_masks_batch
        #     ).sum() / active_masks_batch.sum()
        # else:
        surr = torch.min(surr1, surr2)
        policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()
        self.optimizers["actor"].zero_grad()
        policy_loss = (
            policy_action_loss
            - dist_entropy * self._policy.custom_config["entropy_coef"]
        )
        policy_loss.backward()

        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.actor.parameters(), self.max_grad_norm
            )
        self.optimizers["actor"].step()
        # ============================== Critic optimization ================================
        value_loss = self._calc_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        self.optimizers["critic"].zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.critic.parameters(), self.max_grad_norm
            )
        self.optimizers["critic"].step()
        stats = dict(
            ratio=float(imp_weights.detach().mean().cpu().numpy()),
            ratio_std=float(imp_weights.detach().std().cpu().numpy()),
            policy_loss=float(policy_loss.detach().cpu().numpy()),
            value_loss=float(value_loss.detach().cpu().numpy()),
            entropy=float(dist_entropy.detach().cpu().numpy()),
        )


        return stats

    def _evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        actions_batch,
        available_actions_batch,
        dones_batch,
        active_masks_batch,
    ):
        logits = self._policy.actor(**{EpisodeKey.CUR_OBS: obs_batch})
        logits -= 1e10 * (1 - available_actions_batch)
        dist = torch.distributions.Categorical(logits=logits)
        # TODO(ziyu): check the shape!!!
        action_log_probs = dist.log_prob(
            actions_batch.view(logits.shape[:-1])
        )  # squeeze the last 1 dimension which is just 1
        dist_entropy = dist.entropy().mean()
        values = self._policy.critic(
            **{EpisodeKey.CUR_OBS: share_obs_batch,
               EpisodeKey.ACTION_MASK: available_actions_batch}
        )

        return values, action_log_probs, dist_entropy

    def _calc_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch=None
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if active_masks_batch is not None:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss



    def zero_grad(self):
        pass

    def step(self):
        pass
