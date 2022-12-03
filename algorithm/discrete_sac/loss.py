from tkinter import TRUE
import torch
import torch.nn.functional as F
from utils.episode import EpisodeKey
from algorithm.common.loss_func import LossFunc
from utils.logger import Logger
from registry import registry
from torch.distributions import Categorical, Normal


@registry.registered(registry.LOSS)
class DiscreteSACLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._use_huber_loss = False
        if self._use_huber_loss:
            self.huber_delta = 10.0

        self._use_max_grad_norm = True
        self.use_auto_alpha = False
        self._alpha = 0.2

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

    # def step(self):
    #     self.policy.soft_update(self._params["tau"])

    def setup_optimizers(self, *args, **kwargs):
        if self.optimizers is None:
            optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
            self.optimizers = {
                "actor": optim_cls(
                    self.policy.actor.parameters(), lr=self._params["actor_lr"]
                ),
                "critic_1": optim_cls(
                    self.policy.critic_1.parameters(), lr=self._params["critic_lr"]
                ),
                "critic_2": optim_cls(
                    self.policy.critic_2.parameters(), lr=self._params["critic_lr"]
                ),
            }
            if self.use_auto_alpha:
                self.optimizers["alpha"] = optim_cls(
                    [self.policy._log_alpha], lr=self._params["alpha_lr"]
                )
        else:
            self.optimizers["actor"].param_groups = []
            self.optimizers["actor"].add_param_group(
                {"params": self.policy.actor.parameters()}
            )
            self.optimizers["critic_1"].param_groups = []
            self.optimizers["critic_1"].add_param_group(
                {"params": self.policy.critic_1.parameters()}
            )
            self.optimizers["critic_2"].param_groups = []
            self.optimizers["critic_2"].add_param_group(
                {"params": self.policy.critic_2.parameters()}
            )
            if self.use_auto_alpha:
                self.optimizers["alpha"].param_groups = []
                self.optimizers["alpha"].add_param_group(
                    {"params": [self.policy._log_alpha]}
                )

    def loss_compute(self, sample):
        self.step_ctr += 1
        policy = self._policy
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)
        self.gamma = policy.custom_config["gamma"]
        target_update_freq = policy.custom_config["target_update_freq"]
        target_update_lr = policy.custom_config["target_update_lr"]
        alpha = self._alpha

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

        pred_q_1 = self.policy.critic_1(
            **{EpisodeKey.CUR_OBS: observations, EpisodeKey.ACTION_MASK: action_masks}
        )
        pred_q_1 = pred_q_1.gather(1, actions).flatten()

        pred_q_2 = (
            self.policy.critic_2(
                **{
                    EpisodeKey.CUR_OBS: observations,
                    EpisodeKey.ACTION_MASK: action_masks,
                }
            )
            .gather(1, actions)
            .flatten()
        )
        next_action_logits = self.policy.target_actor(
            **{EpisodeKey.CUR_OBS: observations}
        )
        next_action_dist = Categorical(logits=next_action_logits)
        next_q = next_action_dist.probs * torch.min(
            self.policy.target_critic_1(
                **{
                    EpisodeKey.NEXT_OBS: next_observations,
                    EpisodeKey.NEXT_ACTION_MASK: next_action_masks,
                }
            ),
            self.policy.target_critic_2(
                **{
                    EpisodeKey.NEXT_OBS: next_observations,
                    EpisodeKey.NEXT_ACTION_MASK: next_action_masks,
                }
            ),
        )
        next_q = next_q.sum(dim=-1) + alpha * next_action_dist.entropy()
        target_q = rewards + self.gamma * next_q.detach() * (1.0 - dones)
        critic_loss_1 = (pred_q_1 - target_q).pow(2).mean()
        critic_loss_2 = (pred_q_2 - target_q).pow(2).mean()

        self.optimizers["critic_1"].zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.critic_1.parameters(), self.max_grad_norm
        )
        self.optimizers["critic_1"].step()

        self.optimizers["critic_2"].zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.critic_2.parameters(), self.max_grad_norm
        )
        self.optimizers["critic_2"].step()

        # actor update
        policy_action_logits = self.policy.actor(**{EpisodeKey.CUR_OBS: observations})
        policy_action_dist = Categorical(logits=policy_action_logits)
        policy_entropy = policy_action_dist.entropy()

        with torch.no_grad():
            current_q_1 = self.policy.critic_1(
                **{
                    EpisodeKey.CUR_OBS: observations,
                    EpisodeKey.ACTION_MASK: action_masks,
                }
            )
            current_q_2 = self.policy.critic_2(
                **{
                    EpisodeKey.CUR_OBS: observations,
                    EpisodeKey.ACTION_MASK: action_masks,
                }
            )
            current_q = torch.min(current_q_1, current_q_2)
        actor_loss = -(
            alpha * policy_entropy + (policy_action_dist.probs * current_q).sum(dim=-1)
        ).mean()
        self.optimizers["actor"].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), self.max_grad_norm
        )
        self.optimizers["actor"].step()

        if self.use_auto_alpha:
            log_prob = -policy_entropy.detach() + self.policy._target_entropy
            alpha_loss = -(self.policy._log_alpha * log_prob).mean()
            self.optimizers["alpha"].zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.policy._log_alpha], self.max_grad_norm)
            self.optimizers["alpha"].step()
            self._alpha = self.policy._log_alpha.detach().exp()

        loss_names = [
            "policy_loss",
            "value_loss_1",
            "value_loss_2",
            "alpha_loss",
            "alpha",
        ]

        stats_list = [
            actor_loss.detach().cpu().numpy(),
            critic_loss_1.detach().cpu().numpy(),
            critic_loss_2.detach().cpu().numpy(),
            alpha_loss.detach().cpu().numpy() if self.use_auto_alpha else 0.0,
            self._alpha.numpy() if self.use_auto_alpha else self._alpha,
        ]

        # TODO: update targe networks

        return dict(zip(loss_names, stats_list))
