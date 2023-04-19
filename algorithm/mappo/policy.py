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

from algorithm.common.policy import Policy

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
    offset = len(instance.preprocessor.shape)
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
class MAPPO(Policy):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        self.random_exploration = False
        model_type = model_config.get(
            "model", "gr_football.basic"
        )  # TODO(jh): legacy issue

        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("model.{}".format(model_type))
        self.share_backbone = model.share_backbone
        assert not self.share_backbone, "jh: not supported now, but easy to implement"
        self.feature_encoder = model.FeatureEncoder()

        # jh: re-define observation space based on feature encoder
        observation_space = self.feature_encoder.observation_space
        action_space = self.feature_encoder.action_space

        super(MAPPO, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.opt_cnt = 0
        self.register_state(self.opt_cnt, "opt_cnt")

        self._use_q_head = custom_config["use_q_head"]
        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )
        self.env_agent_id = kwargs["env_agent_id"]

        # # TODO(ming): will collect to custom config
        # global_observation_space = custom_config["global_state_space"][
        #     kwargs["env_agent_id"]
        # ]
        breakpoint()

        actor = model.Actor(
            self.model_config["actor"],
            observation_space,
            action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        # TODO(jh): retrieve from feature encoder as well.
        global_observation_space = observation_space

        critic = model.Critic(
            self.model_config["critic"],
            global_observation_space,
            action_space if self._use_q_head else gym.spaces.Discrete(1),
            self.custom_config,
            self.model_config["initialization"],
        )

        self.observation_space = observation_space
        self.action_space = action_space

        # allow initialize weights in model
        # NOTE(jh): if load later, those parameters will be overwritten.
        # if "pretrained_path" in self.model_config["initialization"]:
        #     Logger.warning("try to initialize from pretrained models")
        #     # TODO: jh should we check desc?
        #     pretrained_path=self.model_config["initialization"]["pretrained_path"]
        #     pretrained_actor_path=os.path.join(pretrained_path,"actor.pt")
        #     if os.path.isfile(pretrained_actor_path):
        #         pretrained_actor = torch.load(pretrained_actor_path,map_location=self.device)
        #         hard_update(actor,pretrained_actor)
        #         Logger.warning("load actor weights from {}".format(pretrained_actor_path))
        #     pretrained_critic_path=os.path.join(pretrained_path,"critic.pt")
        #     if os.path.isfile(pretrained_critic_path):
        #         pretrained_critic = torch.load(pretrained_critic_path,map_location=self.device)
        #         hard_update(critic,pretrained_critic)
        #         Logger.warning("load critic weights from {}".format(pretrained_critic_path))

        # register state handler
        self.set_actor(actor)
        self.set_critic(critic)

        if custom_config["use_popart"]:
            self.value_normalizer = PopArt(
                1, device=self.device, beta=custom_config["popart_beta"]
            )
            self.register_state(self.value_normalizer, "value_normalizer")

        self.register_state(self._actor, "actor")
        self.register_state(self._critic, "critic")

    # TODO(jh): check & add later
    # def reset_layers(self, layer_name: Dict):
    #     use_orthogonal = self.model_config["initialization"]['use_orthogonal']
    #     init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

    #     def init_weights(m):
    #         if type(m) == nn.Linear:
    #             init_fc_weights(m, init_method, self.model_config["initialization"]["gain"])
    #         else:
    #             raise NotImplementedError

    #     network_name = layer_name['network_name'].split(',')   #actor, critic
    #     layer_type = layer_name['layer_type'].split(',')       #base, out
    #     layer_idx = layer_name['layer_idx'].split(',')
    #     if 'actor' in network_name:
    #         if 'out' in layer_type:
    #             self._actor.out.apply(init_weights)

    #         if 'base' in layer_idx:
    #             linear_cnt = 0
    #             total_num_layer = len(self._actor.base.net)
    #             for i in range(total_num_layer):
    #                 if type(self._actor.base.net[i]) == nn.Linear:
    #                     # if linear_cnt == layer_idx:
    #                     if str(linear_cnt) in layer_idx:
    #                         self._actor.base.net[i].apply(init_weights)
    #                         # break
    #                     linear_cnt += 1

    #     if 'critic' in network_name:
    #         if 'out' in layer_type:
    #             self._critic.out.apply(init_weights)
    #         if 'base' in layer_idx:
    #             linear_cnt = 0
    #             total_num_layer = len(self._critic.base.net)
    #             for i in range(total_num_layer):
    #                 if type(self._critic.base.net[i]) == nn.Linear:
    #                     if str(linear_cnt) in layer_idx:
    #                         self._critic.base.net[i].apply(init_weights)
    #                         # break
    #                     linear_cnt += 1

    #     # Logger.warning(f"Reseting {network_name}'s layer")
    #     print(f'reseting {network_name} layer...')

    # def noise_layers(self, layer_name: Dict):

    #     def add_noise(weights, noise_scale):
    #         noise = torch.randn(weights.size())*noise_scale
    #         with torch.no_grad():
    #             weights.add_(noise)

    #     network_name = layer_name['network_name'].split(',')   #actor, critic
    #     layer_type = layer_name['layer_type']       #base, out
    #     layer_idx = layer_name['layer_idx']
    #     n_scale = layer_name['noise_scale']
    #     if 'actor' in network_name:
    #         if layer_type == 'out':     #output layer
    #             add_noise(self._actor.out.weight, noise_scale=n_scale)
    #         elif layer_type == 'base':
    #             linear_cnt = 0
    #             total_num_layer = len(self._actor.base.net)
    #             for i in range(total_num_layer):
    #                 if type(self._actor.base.net[i]) == nn.Linear:
    #                     if linear_cnt == layer_idx:
    #                         add_noise(self._actor.base.net[i].weight, noise_scale=n_scale)
    #                         break
    #                     linear_cnt += 1

    #     if 'critic' in network_name:
    #         if layer_type == 'out':     #output layer
    #             add_noise(self._critic.out.weight, noise_scale=n_scale)
    #         elif layer_type == 'base':
    #             linear_cnt = 0
    #             total_num_layer = len(self._critic.base.net)
    #             for i in range(total_num_layer):
    #                 if type(self._critic.base.net[i]) == nn.Linear:
    #                     if linear_cnt == layer_idx:
    #                         add_noise(self._critic.base.net[i].weight, noise_scale=n_scale)
    #                         break
    #                     linear_cnt += 1

    #     print(f'noising {network_name} layer...')

    def get_initial_state(self, batch_size) -> List[DataTransferType]:
        return {
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(
                (batch_size, self._actor.rnn_layer_num, self._actor.rnn_state_size)
            ),
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(
                (batch_size, self._critic.rnn_layer_num, self._critic.rnn_state_size)
            ),
        }

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy._actor = self_copy._actor.to(device)
        self_copy._critic = self_copy._critic.to(device)
        if self.custom_config["use_popart"]:
            self_copy.value_normalizer = self_copy.value_normalizer.to(device)
            self_copy.value_normalizer.tpdv = dict(dtype=torch.float32, device=device)
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
            actor_rnn_states = kwargs[EpisodeKey.ACTOR_RNN_STATE]
            critic_rnn_states = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            action_masks = kwargs[EpisodeKey.ACTION_MASK]
            rnn_masks = kwargs[EpisodeKey.DONE]

            if hasattr(self.actor, "compute_action"):
                actions, actor_rnn_states, action_probs = self.actor.compute_action(
                    observations, actor_rnn_states, rnn_masks, action_masks
                )
            else:
                logits, actor_rnn_states = self.actor(
                    observations, actor_rnn_states, rnn_masks
                )
                # if "action_mask" in kwargs:
                illegal_action_mask = torch.FloatTensor(1 - action_masks).to(
                    logits.device
                )
                # assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
                #     illegal_action_mask.max(),
                #     illegal_action_mask.min(),
                # )
                logits = logits - 1e10 * illegal_action_mask

                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                action_probs = dist.probs  # num_action

            actor_rnn_states = actor_rnn_states.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            if self.random_exploration:
                exploration_actions = np.zeros(actions.shape, dtype=int)
                for i in range(len(actions)):
                    if random.uniform(0, 1) < self.random_exploration:
                        exploration_actions[i] = int(random.choice(range(19)))
                    else:
                        exploration_actions[i] = int(actions[i])
                actions = exploration_actions

            action_probs = action_probs.detach().cpu().numpy()

            if EpisodeKey.CUR_STATE not in kwargs:
                states = observations

            values, critic_rnn_states = self.critic(
                states, critic_rnn_states, rnn_masks
            )
            values = values.detach().cpu().numpy()
            critic_rnn_states = critic_rnn_states.detach().cpu().numpy()

            return {
                EpisodeKey.ACTION: actions,
                EpisodeKey.ACTION_DIST: action_probs,
                EpisodeKey.STATE_VALUE: values,
                EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states,
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
        torch.save(self._actor, os.path.join(dump_dir, "actor.pt"))
        torch.save(self._critic, os.path.join(dump_dir, "critic.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        res = MAPPO(
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
            hard_update(res._actor, actor)
        if os.path.exists(critic_path):
            critic = torch.load(os.path.join(dump_dir, "critic.pt"), res.device)
            hard_update(res._critic, critic)
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
