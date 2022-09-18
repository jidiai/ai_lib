import threading
from ..utils.desc.task_desc import PrefetchingDesc, RolloutDesc, TrainingDesc
from ..utils.remote import get_actor
from . import distributed_trainer
import ray
from ..utils.decorator import limited_calls
from .data_prefetcher import DataPrefetcher
import numpy as np

class TrainingManager:
    def __init__(self,cfg):
        self.cfg=cfg
        self.rollout_manger=get_actor("TrainingManager","RolloutManager")
        self.data_server=get_actor("TrainingManager","DataServer")
        self.monitor=get_actor("TrainingManager","Monitor")
        
        DistributedTrainer=ray.remote(num_gpus=1,num_cpus=1)(distributed_trainer.DistributedTrainer)
        self.trainers=[
            DistributedTrainer.remote(
                id=self.default_trainer_id(idx),
                local_rank=idx,
                world_size=self.cfg.num_gpus, # TODO(jh) check ray resouces if we have so many gpus.
                master_addr=self.cfg.master_addr,
                master_port=self.cfg.master_port,
                gpu_preload=False, # TODO(jh): debug
                local_queue_size=self.cfg.local_queue_size
            ) for idx in range(self.cfg.num_gpus)
        ]
        self.prefetchers=[
            DataPrefetcher.remote(self.trainers)
            for i in range(self.cfg.num_prefetchers)
        ]
        
        self.stop_flag=True
        self.stop_flag_lock=threading.Lock()
        # cannot start two rollout tasks
        self.semaphore=threading.Semaphore(value=1)
        
    @staticmethod
    def default_trainer_id(idx):
        return "trainer_{}".format(idx)
    
    @limited_calls("semaphore")
    def train(self,training_desc:TrainingDesc):
        with self.stop_flag_lock:
            assert self.stop_flag
            self.stop_flag=False
            
        table_name=ray.get(self.data_server.default_table_name.remote(
            rollout_desc.agent_id,
            rollout_desc.policy_id
        ))
        
        rollout_desc=RolloutDesc(
            training_desc.agent_id,
            training_desc.policy_id,
            training_desc.policy_distributions,
            others={
                "table_name": table_name,
                "max_rollout_epoch": self.cfg.max_rollout_epoch
            }
        )
        rollout_task_ref=self.rollout_manger.rollout.remote(rollout_desc)
        
        prefetching_desc=PrefetchingDesc(
            table_name,
            self.cfg.batch_size
        )
        prefetching_task_refs=[prefetcher.prefetch.remote(prefetching_desc) for prefetcher in self.prefetchers]
        
        training_desc.kwargs["cfg"]=self.cfg.trainer
        ray.get([
            trainer.reset.remote(
                training_desc
            ) for trainer in self.trainers
        ])
                
        self.training_steps=0
        # training process
        while True:
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            self.training_steps+=1   
            
            training_statistics_list=ray.get(
                [
                    trainer.optimize.remote(  
                    ) for trainer in self.trainers
                ]
            )
            
            # reduce
            training_statistics=self.reduce_statistics(training_statistics_list)
            # log
            for key,value in training_statistics.items():
                tag="Training/{}/{}/{}".format(
                    training_desc.agent_id,
                    training_desc.policy_id,
                    key
                )
                self.monitor.add_scalar(tag,value,global_epoch=self.training_steps)
            
            # push new policy
            if self.training_steps%self.cfg.update_interval:
                ray.get(self.trainers[0].push_policy.remote())
        
        # signal prefetchers to stop prefetching
        ray.get([prefetcher.stop_prefetching.remote() for prefetcher in self.prefetchers])
        # wait for prefetching tasks to completely stop
        ray.get(prefetching_task_refs)
        
        # signal rollout_manager to stop rollout
        ray.get(self.rollout_manger.stop_rollout.remote())
        # wait for rollout task to completely stop
        ray.get(rollout_task_ref)
        
    def stop_training(self):
        with self.stop_flag_lock:
            self.stop_flag=True
            
    def reduce_statistics(self,statistics_list):
        statistics={k:[] for k in statistics_list[0]}
        for s in statistics_list:
            for k,v in s.items():
                statistics[k].append(v)
        for k,v in statistics.items():
            # maybe other reduce method
            statistics[k]=np.mean(v)
        return statistics