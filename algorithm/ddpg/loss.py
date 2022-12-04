from tkinter import TRUE
import torch
import torch.nn.functional as F
from utils.episode import EpisodeKey
from algorithm.common.loss_func import LossFunc
from utils.logger import Logger
from registry import registry
from torch.distributions import Categorical, Normal
from ..common.misc import onehot_from_logits, gumbel_softmax


@registry.registered(registry.LOSS)
class DDPGLoss(LossFunc):
    def __init__(self):
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

    def zero_grad(self):
        pass

    def step(self):
        pass

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""

        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "actor": optim_cls(
                    self.policy.actor.parameters(), lr=self._params["actor_lr"]
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
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)
        self.gamma = policy.custom_config["gamma"]

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
        self.optimizers["critic"].zero_grad()
        with torch.no_grad():
            target_action = self.policy.target_actor(
                **{EpisodeKey.NEXT_OBS: next_observations}
            )
            if self.policy.discrete_action:
                target_action = onehot_from_logits(target_action)

        target_vf_in = torch.cat([next_observations, target_action], dim=-1)
        next_value = self.policy.target_critic(**{EpisodeKey.OBS_ACTION: target_vf_in})
        target_value = rewards + self.gamma * next_value * (1.0 - dones)
        vf_in = torch.cat([observations, actions], dim=-1)
        actual_value = self.policy.critic(**{EpisodeKey.OBS_ACTION: vf_in})
        assert actual_value.shape == target_value.shape, (
            actual_value.shape,
            target_value.shape,
            rewards.shape,
        )

        value_loss = torch.nn.MSELoss()(actual_value, target_value.detach())
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(), self.max_grad_norm
        )
        self.optimizers["critic"].step()

        # --------------------------------
        self.optimizers["actor"].zero_grad()

        current_action = self.policy.actor(**{EpisodeKey.CUR_OBS: observations})
        if self.policy.discrete_action:
            current_action = onehot_from_logits(current_action)

        vf_in = torch.cat([observations, current_action], dim=-1)
        policy_loss = -self.policy.critic(
            **{EpisodeKey.OBS_ACTION: vf_in}
        ).mean()  # need add regularization?
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), self.max_grad_norm
        )
        self.optimizers["actor"].step()

        loss_names = ["policy_loss", "value_loss", "target_value_est", "eval_value_est"]

        stats_list = [
            policy_loss.detach().item(),
            value_loss.detach().item(),
            target_value.mean().item(),
            actual_value.mean().item(),
        ]

        return dict(zip(loss_names, stats_list))
