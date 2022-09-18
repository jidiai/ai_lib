from ..desc.policy_desc import PolicyDesc
from light_malib.desc.task_desc import TrainingDesc
from ..utils.remote import get_actor
from malib.utils.logger import Logger
import torch
from torch.nn.parallel import DistributedDataParallel
import os
import ray
from torch import distributed
import queue
from .data_prefetcher import GPUPreLoadQueueWrapper
import numpy as np
from .trainer import Trainer

class DistributedPolicyWrapper:
    '''
    TODO much more functionality
    '''
    def __init__(self,policy,local_rank):
        Logger.info("local_rank: {} cuda_visible_devices:{}".format(local_rank,os.environ["CUDA_VISIBLE_DEVICES"]))
        self.device=torch.device("cuda:0")
        self.policy=policy.to_device(self.device)
        
        actor=self.policy.actor
        self.actor=DistributedDataParallel(actor,device_ids=[0])
        
        critic=self.policy.critic
        self.critic=DistributedDataParallel(critic,device_ids=[0])
        
        # TODO jh: we need a distributed version of value_normalizer
        value_normalizer=self.policy.value_normalizer
        self.value_normalizer=value_normalizer

    @property
    def custom_config(self):
        return self.policy.custom_config
    
    @property
    def opt_cnt(self):
        return self.policy.opt_cnt
    
    @opt_cnt.setter
    def opt_cnt(self,value):
        self.policy.opt_cnt=value

    def evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        actions_batch,
        available_actions_batch,
        actor_rnn_states_batch,
        critic_rnn_states_batch,
        dones_batch,
        active_masks_batch=None,
    ):
        return self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            active_masks_batch
        )
    
class DistributedTrainer:
    def __init__(self,
                 id,
                 local_rank,
                 world_size,
                 master_addr,
                 master_port,
                 gpu_preload,
                 local_queue_size
                 ):
        #Logger.error("distribtued to initialize")
        os.environ["MASTER_ADDR"]=master_addr
        os.environ["MASTER_PORT"]=master_port
        distributed.init_process_group("gloo",rank=local_rank,world_size=world_size)
        
        self.id=id
        self.local_rank=local_rank
        self.world_size=world_size
        self.gpu_preload=gpu_preload
        self.device=torch.device("cuda:0")
        self.trainer=Trainer(self.id,)
        self.cfg=None
 
        self.policy_server=get_actor(self.id,"PolicyServer")

        self.local_queue=queue.Queue(local_queue_size)
        if gpu_preload:
            self.local_queue=GPUPreLoadQueueWrapper(self.local_queue)
            
        Logger.warning("ditributed trainer (local rank: {}) initialized".format(local_rank))
        
    def local_queue_put(self,data):
        if self.gpu_preload:
            data=GPUPreLoadQueueWrapper.to_pin_memory(data)
        try:
            self.local_queue.put(data,block=True,timeout=10)
        except queue.Full:
            Logger.info("queue is full")
    
    def local_queue_get(self):
        data=self.local_queue.get(block=True)
        return data
            
    def reset(self,training_desc:TrainingDesc):
        self.agent_id=training_desc.agent_id
        self.policy_id=training_desc.policy_id
        self.cfg=training_desc.kwargs["cfg"]
        # pull from policy_server
        policy=self.policy_server.pull.remote(self.agent_id,self.policy_id)
        # wrap policies to distributed ones
        self.policy=DistributedPolicyWrapper(policy,self.local_rank)
        self.trainer.reset(self.policy,self.cfg)
                
    def is_main(self):
        return self.local_rank==0
    
    def optimize(self):
        batch=self.local_queue_get()
        training_info=self.trainer.optimize(self.policy,batch)
        if self.is_main():
            self.push_policy()
        return training_info
    
    def get_unwrapped_policy(self):
        return self.policy.policy
    
    def push_policy(self):
        policy_desc=PolicyDesc(
            self.agent_id,
            self.policy_id,
            self.get_unwrapped_policy().to_device("cpu")
        )
        ray.get(self.policy_server.push.remote(policy_desc))
        
    def dump_policy(self):
        pass

    def __del__(self):
        # TODO jh: ?
        distributed.destroy_process_group()