import torch
import numpy as np

from utils.episode import EpisodeKey
from utils.logger import Logger


def compute_return(policy, batch):
    return_mode = policy.custom_config["return_mode"]
    if return_mode == "gae":
        raise NotImplementedError
    elif return_mode == "vtrace":
        raise NotImplementedError
    elif return_mode in ["new_gae", "async_gae"]:
        return compute_async_gae(policy, batch)
    elif return_mode in ["mc"]:
        return compute_mc(policy, batch)
    elif return_mode == 'reward_to_go':
        return reward_to_go(policy, batch)
    else:
        raise ValueError("Unexpected return mode: {}".format(return_mode))


# def compute_gae(policy, batch):
#     with torch.no_grad()
def reward_to_go(policy, batch):
    """
    batch: [B, traj_len, num_agent, _];
    """
    # breakpoint()
    cfg = policy.custom_config
    gamma = cfg["gamma"]
    reward = batch[EpisodeKey.REWARD]
    batch_size, traj_len, num_agent, reward_dim = reward.shape
    Gt = []
    R = np.zeros((batch_size, num_agent, reward_dim))
    for t in reversed(range(traj_len)):
        r = reward[:,t,...]
        R = r + gamma*R
        Gt.insert(0, R)
    Gt = np.swapaxes(np.stack(Gt), 0,1)

    obs= batch[EpisodeKey.CUR_OBS].reshape(batch_size*traj_len*num_agent, -1)
    action_masks = batch[EpisodeKey.ACTION_MASK].reshape(batch_size*traj_len*num_agent, -1)

    device = policy.device
    obs = torch.FloatTensor(obs).to(device)
    action_masks = torch.FloatTensor(action_masks).to(device)

    V = policy.critic(**{EpisodeKey.CUR_OBS: obs,
                         EpisodeKey.ACTION_MASK: action_masks}).detach().cpu().numpy()
    V = V.reshape(batch_size, traj_len, num_agent, -1)
    delta = Gt-V
    adv = delta

    ret = {
        EpisodeKey.RETURN: Gt,
        EpisodeKey.STATE_VALUE: V,
        EpisodeKey.ADVANTAGE: adv
    }
    return ret





def compute_async_gae(policy, batch):
    """
    NOTE the last obs,state,done,critic_rnn_states are for bootstraping.
    """
    with torch.no_grad():
        cfg = policy.custom_config
        gamma, gae_lambda = cfg["gamma"], cfg["gae"]["gae_lambda"]
        rewards = batch[EpisodeKey.REWARD]
        dones = batch[EpisodeKey.DONE]
        cur_obs = batch[EpisodeKey.CUR_OBS]
        rnn_states = batch[EpisodeKey.CRITIC_RNN_STATE]

        # Logger.error("use new gae")
        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert (
            rewards.shape[1] == Tp1 - 1
            and dones.shape[1] == Tp1
            and rnn_states.shape[1] == Tp1
        ), "{}".format({k: v.shape for k, v in batch.items()})

        obs = cur_obs.reshape((B * Tp1 * N, -1))
        rnn_states = rnn_states.reshape((B * Tp1 * N, *rnn_states.shape[-2:]))
        masks = dones.reshape((B * Tp1 * N, -1))

        normalized_value, _ = policy.critic(obs, rnn_states, masks)
        normalized_value = normalized_value.reshape((B, Tp1, N, -1)).detach()

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(
                normalized_value.reshape(-1, normalized_value.shape[-1])
            )
            values = values.reshape(normalized_value.shape)
        else:
            values = normalized_value

        gae = 0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(Tp1 - 1)):
            delta = (
                rewards[:, t]
                + gamma * (1 - dones[:, t]) * values[:, t + 1]
                - values[:, t]
            )
            gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * gae
            # TODO(jh): we should differentiate terminal case and truncation case. now we directly follow env's dones
            # gae *= (1-done[t])          #terminal case
            advantages[:, t] = gae

        returns = advantages + values[:, :-1]

        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(
                returns.reshape(-1, rewards.shape[-1])
            )
            normalized_returns = normalized_returns.reshape(rewards.shape)
        else:
            normalized_returns = returns

        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())

        ret = {
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: normalized_value[:, :-1],
            EpisodeKey.ADVANTAGE: advantages,
        }

        # remove bootstraping data
        for key in [
            EpisodeKey.CUR_OBS,
            EpisodeKey.DONE,
            EpisodeKey.CRITIC_RNN_STATE,
            EpisodeKey.CUR_STATE,
        ]:
            if key in batch:
                ret[key] = batch[key][:, :-1]

        return ret


def compute_mc(policy, batch):
    with torch.no_grad():
        cfg = policy.custom_config
        gamma = cfg["gamma"]
        rewards = batch[EpisodeKey.REWARD]
        dones = batch[EpisodeKey.DONE]
        cur_obs = batch[EpisodeKey.CUR_OBS]
        rnn_states = batch[EpisodeKey.CRITIC_RNN_STATE]

        # Logger.error("use new gae")
        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert (
            rewards.shape[1] == Tp1 - 1
            and dones.shape[1] == Tp1
            and rnn_states.shape[1] == Tp1
        ), "{}".format({k: v.shape for k, v in batch.items()})

        # get last step for boostrapping
        obs = cur_obs.reshape((B * Tp1 * N, -1))
        rnn_states = rnn_states.reshape((B * Tp1 * N, *rnn_states.shape[-2:]))
        masks = dones.reshape((B * Tp1 * N, -1))

        normalized_value, _ = policy.critic(obs, rnn_states, masks)
        normalized_value = normalized_value.reshape((B, Tp1, N, -1)).detach()

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(
                normalized_value.reshape(-1, normalized_value.shape[-1])
            )
            values = values.reshape(normalized_value.shape)
        else:
            values = normalized_value

        ret = 0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(Tp1 - 1)):
            ret = gamma * (1 - dones[:, t]) * ret + rewards[:, t]
            if t == Tp1 - 1 - 1:
                # bootstrapping values
                ret += (1 - dones[:, t]) * values[:, t + 1]
            advantages[:, t] = ret - values[:, t]

        returns = advantages + values[:, :-1]

        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(
                returns.reshape(-1, rewards.shape[-1])
            )
            normalized_returns = normalized_returns.reshape(rewards.shape)
        else:
            normalized_returns = returns

        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())

        ret = {
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: normalized_value[:, :-1],
            EpisodeKey.ADVANTAGE: advantages,
        }

        # remove bootstraping data
        for key in [
            EpisodeKey.CUR_OBS,
            EpisodeKey.DONE,
            EpisodeKey.CRITIC_RNN_STATE,
            EpisodeKey.CUR_STATE,
        ]:
            if key in batch:
                ret[key] = batch[key][:, :-1]

        return ret
