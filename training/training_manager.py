import threading

from utils.naming import default_table_name
from utils.desc.task_desc import PrefetchingDesc, RolloutDesc, TrainingDesc
from utils.distributed import get_actor, get_resources
from . import distributed_trainer
import ray
from utils.decorator import limited_calls
from . import data_prefetcher
import numpy as np
from utils.logger import Logger
from utils.timer import global_timer
from algorithm.mappo.trainer import MAPPOTrainer

class TrainingManager:
    def __init__(self,id,cfg):
        self.id=id
        self.cfg=cfg
        self.rollout_manger=get_actor(self.id,"RolloutManager")
        self.data_server=get_actor(self.id,"DataServer")
        self.policy_server=get_actor(self.id,"PolicyServer")
        self.monitor=get_actor(self.id,"Monitor")
        
        DistributedTrainer=ray.remote(**get_resources(cfg.trainer.distributed.resources))(distributed_trainer.DistributedTrainer)
        DataPrefetcher=ray.remote(**get_resources(cfg.data_prefetcher.distributed.resources))(data_prefetcher.DataPrefetcher)
        
        if self.cfg.master_port is None:
            self.cfg.master_port=str(int(np.random.randint(10000,20000)))
        self.trainers=[
            DistributedTrainer.options(max_concurrency=10).remote(
                id=self.default_trainer_id(idx),
                trainer_cls=MAPPOTrainer,
                local_rank=idx,
                world_size=self.cfg.num_trainers, # TODO(jh) check ray resouces if we have so many gpus.
                master_addr=self.cfg.master_addr,
                master_port=self.cfg.master_port,
                master_ifname=self.cfg.get("master_ifname",None),
                gpu_preload=self.cfg.gpu_preload, # TODO(jh): debug
                local_queue_size=self.cfg.local_queue_size,
                policy_server=self.policy_server
            ) for idx in range(self.cfg.num_trainers)
        ]
        self.prefetchers=[
            DataPrefetcher.options(max_concurrency=10).remote(
                self.cfg.data_prefetcher,
                self.trainers,
                [
                    self.data_server
                ]
            )
            for i in range(self.cfg.num_prefetchers)
        ]
        
        self.stop_flag=True
        self.stop_flag_lock=threading.Lock()
        # cannot start two rollout tasks
        self.semaphore=threading.Semaphore(value=1)
        Logger.info("{} initialized".format(self.id))
        
    @staticmethod
    def default_trainer_id(idx):
        return "trainer_{}".format(idx)
    
    @limited_calls("semaphore")
    def train(self,training_desc:TrainingDesc):
        with self.stop_flag_lock:
            assert self.stop_flag
            self.stop_flag=False
            
        # create table
        table_name=default_table_name(training_desc.agent_id,training_desc.policy_id,training_desc.share_policies)
        ray.get(self.data_server.create_table.remote(table_name))

        rollout_desc=RolloutDesc(
            training_desc.agent_id,
            training_desc.policy_id,
            training_desc.policy_distributions,
            training_desc.share_policies
        )
        rollout_task_ref=self.rollout_manger.rollout.remote(rollout_desc)
                
        training_desc.kwargs["cfg"]=self.cfg.trainer
        ray.get([
            trainer.reset.remote(
                training_desc
            ) for trainer in self.trainers
        ])
        
        prefetching_desc=PrefetchingDesc(
            table_name,
            self.cfg.batch_size
        )
        prefetching_descs=[prefetching_desc]
        prefetching_task_refs=[prefetcher.prefetch.remote(prefetching_descs) for prefetcher in self.prefetchers]
        
        if self.cfg.gpu_preload:
            ray.get([trainer.local_queue_init.remote() for trainer in self.trainers])

        training_steps=0
        # training process
        while True:
            global_timer.record("train_step_start")
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            training_steps+=1   
            
            global_timer.record("optimize_start")
            statistics_list=ray.get(
                [
                    trainer.optimize.remote(  
                    ) for trainer in self.trainers
                ]
            )
            global_timer.time("optimize_start","optimize_end","optimize")
            
            # push new policy
            if training_steps%self.cfg.update_interval==0:
                global_timer.record("push_policy_start")
                ray.get(self.trainers[0].push_policy.remote(training_steps))
                global_timer.time("push_policy_start","push_policy_end","push_policy")
            global_timer.time("train_step_start","train_step_end","train_step")
            
            # reduce
            training_statistics=self.reduce_statistics([statistics[0] for statistics in statistics_list])
            timer_statistics=self.reduce_statistics([statistics[1] for statistics in statistics_list])
            timer_statistics.update(global_timer.elapses)
            
            # log
            log_tasks=[]
            for key,value in training_statistics.items():
                tag="Training/{}/{}/{}".format(
                    training_desc.agent_id,
                    training_desc.policy_id,
                    key
                )
                log_task=self.monitor.add_scalar.remote(tag,value,global_step=training_steps)
                log_tasks.append(log_task)
            for key,value in timer_statistics.items():
                tag="TrainingTimer/{}/{}/{}".format(
                    training_desc.agent_id,
                    training_desc.policy_id,
                    key
                )
                log_task=self.monitor.add_scalar.remote(tag,value,global_step=training_steps)
                log_tasks.append(log_task)
            data_server_statistics = ray.get(self.data_server.get_statistics.remote(table_name))
            for key,value in data_server_statistics.items():
                tag=f"DataServer/{training_desc.agent_id}/{training_desc.policy_id}/{key}"
                log_task=self.monitor.add_scalar.remote(tag,value,global_step=training_steps)
                log_tasks.append(log_task)
            ray.get(log_tasks)
            # Logger.info("Traning {}: {}".format(training_steps,training_statistics))
            
            global_timer.clear()
        
        # signal prefetchers to stop prefetching
        ray.get([prefetcher.stop_prefetching.remote() for prefetcher in self.prefetchers])
        # wait for prefetching tasks to completely stop
        ray.get(prefetching_task_refs)
        
        # signal rollout_manager to stop rollout
        ray.get(self.rollout_manger.stop_rollout.remote())
        # wait for rollout task to completely stop
        ray.get(rollout_task_ref)
        
        # remove table
        ray.get(self.data_server.remove_table.remote(training_desc.agent_id,training_desc.policy_id))
        
        Logger.warning("Training ends after {} steps".format(training_steps))
        
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
    
    def close(self):
        ray.get([trainer.close.remote() for trainer in self.trainers])