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

import wrapt
import tree
import importlib
from utils.logger import Logger
from gym.spaces import Discrete
from ..utils import PopArt
from registry import registry
from copy import deepcopy

from .agent_q_function import AgentQFunction
from torch.distributions import Categorical, OneHotCategorical

from gym.spaces.discrete import Discrete

def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input
def to_numpy(x):
    return x.detach().cpu().numpy()
def avail_choose(x, avail_x=None):
    x = to_torch(x)
    if avail_x is not None:
        avail_x = to_torch(avail_x)
        x[avail_x == 0] = -1e10
    return x#FixedCategorical(logits=x)
def make_onehot(int_action, action_dim, seq_len=None):
    if type(int_action) == torch.Tensor:
        int_action = int_action.cpu().numpy()
    if not seq_len:
        return np.eye(action_dim)[int_action]
    if seq_len:
        onehot_actions = []
        for i in range(seq_len):
            onehot_action = np.eye(action_dim)[int_action[i]]
            onehot_actions.append(onehot_action)
        return np.stack(onehot_actions)


class ExploreActor(nn.Module):
    def __init__(
            self,
            model_config,
            observation_space,
            action_space,
            custom_config,
            initialization,
    ):
        super().__init__()

    def forward(self, **kwargs):
        pass


def hard_update(target, source):
    """Copy network parameters from source to target.

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15

    :param torch.nn.Module target: Net to copy parameters to.
    :param torch.nn.Module source: Net whose parameters to copy
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

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


@registry.registered(registry.POLICY)
class QMix(nn.Module):

    def __init__(
            self,
            registered_name: str,
            observation_space: gym.spaces.Space,  # legacy
            action_space: gym.spaces.Space,  # legacy
            model_config: Dict[str, Any] = None,
            custom_config: Dict[str, Any] = None,
            **kwargs,
    ):
        """
        QMIX/VDN Policy Class to compute Q-values and actions. See parent class for details.
        :param config: (dict) contains information about hyperparameters and algorithm configuration
        :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
        :param train: (bool) whether the policy will be trained.
        """

        self.registered_name = registered_name

        model_type = model_config["model"]
        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("model.{}".format(model_type))
        model = importlib.import_module("model.{}".format(model_type))
        self.encoder = model.Encoder()
        self._rewarder = model.Rewarder()

        self.observation_space = self.feature_encoder.observation_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_space = self.encoder.action_space
        self.act_dim = self.action_space.n
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.hidden_size = 64
        self.central_obs_dim = 115 #policy_config["cent_obs_dim"]
        self.discrete = True
        self.multidiscrete = False
        self.prev_act_inp = False #custom_config.prev_act_inp
        self.rnn_layer_num = 1 #custom_config.local_q_config.layer_N
        assert self.rnn_layer_num == 1, print('only support one rnn layer number')

        self.model_config=  model_config
        self.custom_config = custom_config
        # super(QMix, self).__init__(
        #     registered_name=registered_name,
        #     observation_space=observation_space,
        #     action_space=action_space,
        #     model_config=model_config,
        #     custom_config=custom_config,
        # )
        super().__init__()

        if self.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.q_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.q_network_input_dim = self.obs_dim

        # Local recurrent q network for the agent
        self.n_agent = custom_config.local_q_config.n_agent
        if self.n_agent == 1:
            self.critic = AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)
            self.target_critic = AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)

        else:
            self.critic = [AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)
                           for _ in range(self.n_agent)]
            self.target_critic = [AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)
                           for _ in range(self.n_agent)]
            for i in range(len(self.critic)):
                self.register_module(f'critic_{i}', self.critic[i])
                self.register_module(f"target_critic_{i}", self.target_critic[i])
                hard_update(self.target_critic[i], self.critic[i])

        # self.critic2 = AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)
        self.fake_actor = ExploreActor(model_config["actor"],
                                  observation_space,
                                  action_space,
                                  custom_config,
                                  model_config["initialization"],)

        self.exploration = DecayThenFlatSchedule(self.custom_config.epsilon_start,
                                                 self.custom_config.epsilon_finish,
                                                 self.custom_config.epsilon_anneal_time,
                                                 decay='linear')
        self.current_eps = 0

        # self.critic = AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)
        # self.target_critic = AgentQFunction(custom_config.local_q_config, self.q_network_input_dim, self.act_dim)
        # hard_update(self.target_critic, self.critic)
        # for i,j in self.named_parameters():
        #     Logger.error(f"{i}, {j.shape}")


        # self.epsilon_start = 0.5
        # self.epsilon_finish = 0.01
        # self.epsilon_anneal_time = 1
        # self.exploration = DecayThenFlatSchedule(self.epsilon_start, self.epsilon_finish,
        #                                          self.args.epsilon_anneal_time,
        #                                          decay="linear")
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

    def soft_update(self, tau):
        if self.n_agent!=1:
            for i in range(len(self.critic)):
                soft_update(self.target_critic[i], self.critic[i], tau)
        else:
            soft_update(self.target_critic, self.critic, tau)


    @property
    def feature_encoder(self):  # legacy
        return self.encoder

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.to(device)
        self_copy.device = device
        return self_copy

    def get_initial_state(self, batch_size):

        return {
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(
                (self.rnn_layer_num, batch_size, self.hidden_size), dtype=np.float32
            ),
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(
                (self.rnn_layer_num, batch_size, self.hidden_size), dtype=np.float32
            ),
        }




    def get_q_values(self, obs_batch, prev_action_batch, rnn_states, action_batch=None):
        """
        Computes q values using the given information.
        :param obs: (np.ndarray) agent observations from which to compute q values
        :param prev_actions: (np.ndarray) agent previous actions which are optionally an input to q network
        :param rnn_states: (np.ndarray) RNN states of q network
        :param action_batch: (np.ndarray) if not None, then only return the q values corresponding to actions in action_batch
        :return q_values: (torch.Tensor) computed q values
        :return new_rnn_states: (torch.Tensor) updated RNN states
        """

        # combine previous action with observation for input into q, if specified in args
        if self.prev_act_inp:
            prev_action_batch = to_torch(prev_action_batch)
            input_batch = torch.cat((obs_batch, prev_action_batch), dim=-1)
        else:
            input_batch = obs_batch

        if self.n_agent==1:
            # rnn_states = np.swapaxes(rnn_states, 0,1)
            q_batch,new_rnn_states = self.critic(input_batch, rnn_states)       #TODO: check input-batch shape and new_rnn_states shape
            new_rnn_states = new_rnn_states.unsqueeze(0)
        else:
            q_batch, new_rnn_states = [], []
            for i in range(self.n_agent):
                _q_batch, _new_rnn_states = self.critic[i](input_batch[i][np.newaxis,...], rnn_states[:,i,...][np.newaxis])
                q_batch.append(_q_batch)
                new_rnn_states.append(_new_rnn_states)
            q_batch = torch.concatenate(q_batch)
            new_rnn_states = torch.stack(new_rnn_states).permute(1,0,2)


        if action_batch is not None:
            action_batch = to_torch(action_batch).to(self.device)
            q_values = self.q_values_from_actions(q_batch, action_batch)
        else:
            q_values = q_batch
        return q_values, new_rnn_states

    def q_values_from_actions(self, q_batch, action_batch):
        """
        Get q values corresponding to actions.
        :param q_batch: (torch.Tensor) q values corresponding to every action.
        :param action_batch: (torch.Tensor) actions taken by the agent.
        :return q_values: (torch.Tensor) q values in q_batch corresponding to actions in action_batch
        """
        raise NotImplementedError
        if self.multidiscrete:
            ind = 0
            all_q_values = []
            for i in range(len(self.act_dim)):
                curr_q_batch = q_batch[i]
                curr_action_portion = action_batch[:, :, ind: ind + self.act_dim[i]]
                curr_action_inds = curr_action_portion.max(dim=-1)[1]
                curr_q_values = torch.gather(curr_q_batch, 2, curr_action_inds.unsqueeze(dim=-1))
                all_q_values.append(curr_q_values)
                ind += self.act_dim[i]
            q_values = torch.cat(all_q_values, dim=-1)
        else:
            # convert one-hot action batch to index tensors to gather the q values corresponding to the actions taken
            action_batch = action_batch.max(dim=-1)[1]
            # import pdb; pdb.set_trace()
            q_values = torch.gather(q_batch, 2, action_batch.unsqueeze(dim=-1))

            # q_values is a column vector containing q values for the actions specified by action_batch
        return q_values

    def get_actions(self, obs, prev_actions, rnn_states, available_actions=None, t_env=None, explore=False):
        """See parent class."""
        raise NotImplementedError("Deprecated")
        q_values_out, new_rnn_states = self.get_q_values(obs, prev_actions, rnn_states)
        onehot_actions, greedy_Qs = self.actions_from_q(q_values_out, available_actions=available_actions,
                                                        explore=explore, t_env=t_env)

        return onehot_actions, new_rnn_states, greedy_Qs

    def compute_action(self, **kwargs):
        local_obs = kwargs[EpisodeKey.CUR_OBS]
        action_masks = kwargs[EpisodeKey.ACTION_MASK]
        rollout_step = kwargs['step']

        prev_actions = None
        rnn_states = kwargs[EpisodeKey.CRITIC_RNN_STATE]
        q_values_out, new_rnn_states = self.get_q_values(local_obs, prev_actions, rnn_states)
        onehot_actions, greedy_Qs = self.actions_from_q(q_values_out, available_actions=action_masks,
                                                        explore=kwargs['explore'], t_env=rollout_step)
        action = [np.where(i==1)[0] for i in onehot_actions]
        action = np.concatenate(action)
        return {EpisodeKey.ACTION: action,
                EpisodeKey.CRITIC_RNN_STATE: new_rnn_states.detach().cpu().numpy(),
                EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE]}

    def compute_actions(
        self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        pass

    def actions_from_q(self, q_values, t_env, available_actions=None, explore=False):
        """
        Computes actions to take given q values.
        :param q_values: (torch.Tensor) agent observations from which to compute q values
        :param available_actions: (np.ndarray) actions available to take (None if all actions available)
        :param explore: (bool) whether to use eps-greedy exploration
        :param t_env: (int) env step at which this function was called; used to compute eps for eps-greedy
        :return onehot_actions: (np.ndarray) actions to take (onehot)
        :return greedy_Qs: (torch.Tensor) q values corresponding to greedy actions.
        """
        if self.multidiscrete:
            no_sequence = len(q_values[0].shape) == 2
            batch_size = q_values[0].shape[0] if no_sequence else q_values[0].shape[1]
            seq_len = None if no_sequence else q_values[0].shape[0]
        else:
            no_sequence = len(q_values.shape) == 2
            batch_size = q_values.shape[0] if no_sequence else q_values.shape[1]
            seq_len = None if no_sequence else q_values.shape[0]

        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            q_values = q_values.clone()
            q_values = avail_choose(q_values, available_actions)
        else:
            q_values = q_values

        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)

                if explore:
                    assert no_sequence, "Can only explore on non-sequences"
                    eps = self.exploration.eval(t_env)
                    rand_number = np.random.rand(batch_size)
                    # random actions sample uniformly from action space
                    random_action = Categorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy()
                    take_random = (rand_number < eps).astype(int)
                    action = (1 - take_random) * to_numpy(greedy_action) + take_random * random_action
                    onehot_action = make_onehot(action, self.act_dim[i])
                else:
                    greedy_Q = greedy_Q.unsqueeze(-1)
                    if no_sequence:
                        onehot_action = make_onehot(greedy_action, self.act_dim[i])
                    else:
                        onehot_action = make_onehot(greedy_action, self.act_dim[i], seq_len=seq_len)

                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)

            onehot_actions = np.concatenate(onehot_actions, axis=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                assert no_sequence, "Can only explore on non-sequences"
                eps = self.exploration.eval(t_env)
                self.current_eps = eps
                rand_numbers = np.random.rand(batch_size)
                # random actions sample uniformly from action space
                logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                random_actions = Categorical(logits=logits).sample().numpy()
                take_random = (rand_numbers < eps).astype(int)
                actions = (1 - take_random) * to_numpy(greedy_actions) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                greedy_Qs = greedy_Qs.unsqueeze(-1)
                if no_sequence:
                    onehot_actions = make_onehot(greedy_actions, self.act_dim)
                else:
                    onehot_actions = make_onehot(greedy_actions, self.act_dim, seq_len=seq_len)

        return onehot_actions, greedy_Qs

    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]

        if self.multidiscrete:
            random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i
                              in
                              range(len(self.act_dim))]
            random_actions = np.concatenate(random_actions, axis=-1)
        else:
            if available_actions is not None:
                logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                random_actions = OneHotCategorical(logits=logits).sample().numpy()
            else:
                random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()

        return random_actions

    def init_hidden(self, num_agents, batch_size):
        """See parent class."""
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(num_agents, batch_size, self.hidden_size)

    # def parameters(self):
    #     """See parent class."""
    #     return self.critic.parameters()

    def dump(self, dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
        if self.n_agent==1:
            torch.save(self.critic.state_dict(), os.path.join(dump_dir, "critic_state_dict.pt"))
        else:
            model_dict = {}
            for i in range(self.n_agent):
                model_dict[f"critic_{i}"] = self.critic[i].state_dict()
            torch.save(model_dict, os.path.join(dump_dir, "critic_state_dict.pt"))

        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        policy = QMix(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )

        critic_path = os.path.join(dump_dir, "critic_state_dict.pt")
        if os.path.exists(critic_path):
            critic_state_dict = torch.load(os.path.join(dump_dir,"critic_state_dict.pt"), 'cpu')
            if isinstance(policy.critic, list):
                for idx, critic_name in enumerate(list(critic_state_dict.keys())):
                    policy.critic[idx].load_state_dict(critic_state_dict[critic_name])
                    policy.target_critic[idx] = deepcopy(policy.critic[idx])
            else:
                policy.critic.load_state_dict(critic_state_dict)
                policy.target_critic = deepcopy(policy.critic)
        return policy



class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / \
                np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            ret = max(self.finish, self.start - self.delta * T)
            return ret
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass
