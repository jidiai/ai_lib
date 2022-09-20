from light_malib.algorithm.mappo.actor_critic import RNNNet
from . import encoder_basic

Actor=RNNNet
Critic=RNNNet
share_backbone=False
FeatureEncoder=encoder_basic.FeatureEncoder 