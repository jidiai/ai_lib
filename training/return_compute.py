import torch
import numpy as np

def compute_return(policy, batch):
    return_mode=policy.custom_config["return_mode"]
    if return_mode == "gae":
        raise NotImplementedError
    elif return_mode == "vtrace":
        raise NotImplementedError
    elif return_mode == 'async_gae':
        return compute_async_gae(policy, batch)
    else:
        raise ValueError("Unexpected return mode: {}".format(return_mode))

def compute_async_gae(reward, done, gamma, gae_lambda, policy, rnn_states, cur_obs, cfg):
    '''
    TODO(jh): NOTE win reward is not supported now.
    
    '''
    cm_cfg = policy.custom_config
    gamma, gae_lambda = cm_cfg["gamma"], cm_cfg["gae"]["gae_lambda"]
    
    # Logger.error("use new gae")
    assert len(reward.shape) == 4, (reward.shape, done.shape)
    B, Tp1, N, _ = reward.shape
    dones = np.transpose(done, (1, 0, 2, 3))
    reward = np.transpose(reward, (1, 0, 2, 3))

    _cast = lambda x: torch.tensor(
        x.reshape((B * Tp1 * N, -1)), dtype=torch.float32, device=policy.device
    )
    obs = _cast(cur_obs)
    rnn_states = torch.tensor(
        rnn_states.reshape((B * Tp1 * N, *rnn_states.shape[-2:])),
        dtype=torch.float32,
        device=policy.device,
    )
    masks = _cast(done)

    origin_value, _ = policy.critic(
        obs, rnn_states, masks
    )
    origin_value = origin_value.reshape((B, Tp1, N, -1)).detach().cpu().numpy()
    if cfg["use_popart"]:
        values = policy.value_normalizer.denormalize(origin_value)
    else:
        values = origin_value


    # values = values.reshape((B, Tp1, N, -1)).detach().cpu().numpy()
    value = np.transpose(values, (1, 0, 2, 3))

    gae, ret = 0, np.zeros_like(reward)
    # for t in reversed(range(Tp1)):
    #     if t == Tp1-1:
    #         delta = reward[t] - value[t]  #terminal reward
    #         gae = delta
    #         ret[t] = gae + value[t]
    #     else:
    #         delta = reward[t] + gamma * (1 - dones[t]) * value[t + 1] - value[t]
    #         gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
    #         # gae *= (1-done[t])          #terminal case
    #         ret[t] = gae + value[t]
    for t in reversed(range(Tp1-1)):
        delta = reward[t] + gamma * (1 - dones[t]) * value[t + 1] - value[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        # gae *= (1-done[t])          #terminal case
        ret[t] = gae + value[t]

    return {"return": ret.transpose((1, 0, 2, 3)),
            "value": origin_value}
