import torch
import torch.nn as nn

from algorithm.common.rnn_net import RNNNet
from . import encoder_basic
from gym.spaces import Discrete

class Actor(RNNNet):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        if observation_space is None:
            observation_space=encoder_basic.FeatureEncoder.observation_space
        if action_space is None:
            action_space=Discrete(19)
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)

class Critic(RNNNet):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        if observation_space is None:
            observation_space=encoder_basic.FeatureEncoder.observation_space
        if action_space is None:
            action_space=Discrete(1)
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)

share_backbone=False
FeatureEncoder=encoder_basic.FeatureEncoder