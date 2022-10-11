# -*- coding: utf-8 -*-
from xml.dom.domreg import registered
import torch
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

def to_value(tensor:torch.Tensor):
    return tensor.detach().cpu().item()

def basic_stats(name,tensor:torch.Tensor):
    stats={}
    stats["{}_max".format(name)]=to_value(tensor.max())
    stats["{}_min".format(name)]=to_value(tensor.min())
    stats["{}_mean".format(name)]=to_value(tensor.mean())
    stats["{}_std".format(name)]=to_value(tensor.std())
    return stats

@registry.registered(registry.LOSS)
class MAPPOLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(MAPPOLoss, self).__init__()

        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        if self._use_huber_loss:
            self.huber_delta = 10.0
        # self._use_value_active_masks = False
        # self._use_policy_active_masks = False

        self._use_max_grad_norm = True
        
        # the following are useless now        
        self.inner_clip_param = 0.1
        self.use_modified_mappo = False
        self.use_inner_clip = False
        # TODO double clipping Tencent
        self.use_double_clip = False
        self.double_clip_param = 3

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


    def pre_update_v(self, sample):
        raise NotImplementedError("TODO(jh): check")
        policy=self._policy
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

        self.use_modified_mappo = policy.custom_config.get("use_modified_mappo", False)

        n_agent = 4
        self._policy.opt_cnt += 1
        # cast = lambda x: torch.FloatTensor(x.copy()).to(self._policy.device)
        (
            share_obs_batch,
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_probs_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
        ) = (
            sample[EpisodeKey.CUR_STATE],
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.RETURN],
            sample[EpisodeKey.ACTIVE_MASK],
            sample[EpisodeKey.ACTION_DIST],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTOR_RNN_STATE],
            sample[EpisodeKey.CRITIC_RNN_STATE],
            sample[EpisodeKey.DONE],
        )

        values, action_log_probs, dist_entropy = self._evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            active_masks_batch,
        )
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



    def loss_compute(self, sample):
        policy=self._policy
        self.clip_param = policy.custom_config.get("clip_param",0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm",10)
        
        self.use_modified_mappo = policy.custom_config.get("use_modified_mappo",False)
        
        n_agent=4
        self._policy.opt_cnt += 1
        # cast = lambda x: torch.FloatTensor(x.copy()).to(self._policy.device)
        (
            
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_probs_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            adv_targ
        ) = (
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.RETURN],
            sample.get(EpisodeKey.ACTIVE_MASK,None),
            sample[EpisodeKey.ACTION_DIST],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTOR_RNN_STATE],
            sample[EpisodeKey.CRITIC_RNN_STATE],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.ADVANTAGE]
        )
        if EpisodeKey.CUR_STATE in sample:
            share_obs_batch = sample[EpisodeKey.CUR_STATE]
        else:
            share_obs_batch = sample[EpisodeKey.CUR_OBS]

        values, action_log_probs, dist_entropy = self._evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            active_masks_batch,
        )
        # print(old_action_probs_batch.shape, actions_batch.shape)
        actions_batch=actions_batch.reshape(actions_batch.shape[0],1)
        old_action_log_probs_batch = torch.log(
            old_action_probs_batch.gather(-1, actions_batch)
        )
        imp_weights = torch.exp(
            action_log_probs.unsqueeze(-1) - old_action_log_probs_batch
        )

        if self.use_modified_mappo:
            if self.use_inner_clip:
                o_imp_weights=imp_weights+1e-9*(imp_weights==0)
            # #env*#agent
            imp_weights=imp_weights.view(-1,n_agent)
            batch_size,n_agent=imp_weights.shape
            imp_weights=torch.prod(imp_weights,dim=-1,keepdim=True)
            #print("imp.shape1.5",imp_weights.shape)
            imp_weights=torch.tile(imp_weights,(1,n_agent))            
            #print("imp.shape2",imp_weights.shape)
            imp_weights=imp_weights.view(batch_size*n_agent,1)
            #print("imp.shape3",imp_weights.shape,imp_weights)
            if self.use_inner_clip:
                imp_weights/=o_imp_weights
                imp_weights=torch.clamp(imp_weights,1.0-self.inner_clip_param,1.0+self.inner_clip_param)
                imp_weights*=o_imp_weights
        
        #Logger.info("imp_weights.shape after: {}".format(imp_weights.shape))

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_double_clip:
            surr3=self.double_clip_param*adv_targ

        if active_masks_batch is not None:
            surr=torch.min(surr1,surr2)
            if self.use_double_clip:
                surr=torch.max(surr,surr3)
            policy_action_loss = (
                -torch.sum(surr, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            surr=torch.min(surr1,surr2)
            if self.use_double_clip:
                mask=(adv_targ<0).float()
                surr=torch.max(surr,surr3)*mask+surr*(1-mask)
            policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True
            ).mean()

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

        stats=dict(
            ratio=float(imp_weights.detach().mean().cpu().numpy()),
            ratio_std=float(imp_weights.detach().std().cpu().numpy()),
            policy_loss=float(policy_loss.detach().cpu().numpy()),
            value_loss=float(value_loss.detach().cpu().numpy()),
            entropy=float(dist_entropy.detach().cpu().numpy()),
        )
        
        stats.update(basic_stats("imp_weights",imp_weights))
        stats.update(basic_stats("advantages",adv_targ))
        
        stats["upper_clip_ratio"]=to_value((imp_weights>(1+self.clip_param)).float().mean())
        stats["lower_clip_ratio"]=to_value((imp_weights<(1-self.clip_param)).float().mean())
        stats["clip_ratio"]=stats["upper_clip_ratio"]+stats["lower_clip_ratio"]
        return stats

    def _evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        actions_batch,
        available_actions_batch,
        actor_rnn_states_batch,
        critic_rnn_states_batch,
        dones_batch,
        active_masks_batch
    ):

        logits, _ = self._policy.actor(obs_batch, actor_rnn_states_batch, dones_batch)
        logits -= 1e10 * (1 - available_actions_batch)

        dist = torch.distributions.Categorical(logits=logits)
        # TODO(ziyu): check the shape!!!
        action_log_probs = dist.log_prob(
            actions_batch.view(logits.shape[:-1])
        )  # squeeze the last 1 dimension which is just 1
        dist_entropy = dist.entropy().mean()

        values, _ = self._policy.critic(
            share_obs_batch, critic_rnn_states_batch, dones_batch
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
