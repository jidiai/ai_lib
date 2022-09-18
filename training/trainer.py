# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.mappo.data_generator import (
    recurrent_generator,
    simple_data_generator
)
from malib.algorithm.mappo.loss import MAPPOLoss
from malib.utils.episode import EpisodeKey
import torch
import functools
from malib.utils.logger import Logger
from malib.utils.timer import global_timer
from .return_compute import compute_return

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class Trainer:
    def __init__(self, id):
        self.id=id
        # TODO(jh)
        self.loss = MAPPOLoss()
        
    def reset(self, policy, training_config):
        """Reset policy, called before optimize, and read training configuration"""
        self.policy = policy
        self.training_config=training_config
        if self.loss is not None:
            self.loss.reset(policy, training_config)
        # else:
        #     raise ValueError("Loss has not been initialized yet.")

    def pre_update(self, batch, policy):
        num_mini_batch = 1
        pre_update_epoch = 5
        if policy.custom_config["use_rnn"]:
            data_generator_fn = functools.partial(
                recurrent_generator,
                batch,
                num_mini_batch,
                policy.custom_config["rnn_data_chunk_length"],
                policy.device,
            )
        else:
            data_generator_fn = functools.partial(
                simple_data_generator, batch, num_mini_batch, policy.device
            )
        mini_batch = next(data_generator_fn())
        for i_epoch in range(pre_update_epoch):
            self.loss.pre_update_v(mini_batch)

    def optimize(self, batch, **kwargs):
        total_opt_result = defaultdict(lambda: 0)
        policy = self.loss.policy
        
        # TODO(jh)
        # bootstrap_value = batch.pop('bootstrap_value')
        
        # move data to gpu
        for key,value in batch.items():
            if isinstance(value,np.ndarray):
                value=torch.from_numpy(value)
            batch[key]=value.to(policy.device)
        
        global_timer.record("compute_return_start")
        # TODO(jh)
        if policy.custom_config.get('pre_update_v', False):
            self.pre_update(batch, policy)      #update V in advance and compute GAE with new V
        new_data = compute_return(policy, batch)
        batch.update(new_data)
        global_timer.time("compute_return_start","compute_return_end","compute_return")

        ppo_epoch = policy.custom_config["ppo_epoch"]
        num_mini_batch = policy.custom_config["num_mini_batch"]  # num_mini_batch
        num_updates = num_mini_batch * ppo_epoch

        if policy.custom_config["use_rnn"]:
            data_generator_fn = functools.partial(
                recurrent_generator,
                batch,
                num_mini_batch,
                policy.custom_config["rnn_data_chunk_length"],
                policy.device,
            )
        else:
            data_generator_fn = functools.partial(
                simple_data_generator, batch, num_mini_batch, policy.device
            )

        # jh: special optimization
        if num_mini_batch==1:
            global_timer.record("data_generator_start")
            mini_batch=next(data_generator_fn())
            global_timer.time("data_generator_start","data_generator_end","data_generator")
            for i_epoch in range(ppo_epoch):
                global_timer.record("loss_start")
                tmp_opt_result = self.loss(mini_batch)
                global_timer.time("loss_start","loss_end","loss")
                for k, v in tmp_opt_result.items():
                    total_opt_result[k] = v
        else:
            for i_epoch in range(ppo_epoch):
                for mini_batch in data_generator_fn():
                    global_timer.record("loss_start")
                    tmp_opt_result = self.loss(mini_batch)
                    global_timer.time("loss_start","loss_end","loss")
                    for k, v in tmp_opt_result.items():
                        total_opt_result[k] = v

        # TODO(ziyu & ming): find a way for customize optimizer and scheduler
        #  but now it doesn't affect the performance ...

        # TODO(jh)
        # if kwargs["lr_decay"]:
        #     epoch = kwargs["rollout_epoch"]
        #     total_epoch = kwargs["lr_decay_epoch"]
        #     assert total_epoch is not None
        #     update_linear_schedule(
        #         self.loss.optimizers["critic"],
        #         epoch,
        #         total_epoch,
        #         self.loss._params["critic_lr"],
        #     )
        #     update_linear_schedule(
        #         self.loss.optimizers["actor"],
        #         epoch,
        #         total_epoch,
        #         self.loss._params["actor_lr"],
        #     )


        return total_opt_result

    def preprocess(self, batch, **kwargs):
        pass