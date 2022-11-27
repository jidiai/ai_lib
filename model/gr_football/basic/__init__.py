from algorithm.common.rnn_net import RNNNet
from . import encoder_basic

Actor = RNNNet
Critic = RNNNet
share_backbone = False
FeatureEncoder = encoder_basic.FeatureEncoder
