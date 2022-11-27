# -*- coding: utf-8 -*-
from tkinter import TRUE
import torch
import torch.nn.functional as F
from utils.episode import EpisodeKey
from algorithm.common.loss_func import LossFunc
from utils.logger import Logger
from registry import registry



@registry.registered(registry.LOSS)
class DDPGLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super().__init__()

