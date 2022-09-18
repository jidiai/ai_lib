from collections import defaultdict
from email import policy
import itertools
import os
import threading
from typing import Dict
import numpy as np

from tools.desc.policy_desc import PolicyDesc
from tools.utils.logger import Logger

from tools.remote import get_actor
from ..agent.agent import Agent
from . import rollout_worker
import ray
from omegaconf import DictConfig
import queue
from tools.desc.task_desc import RolloutDesc, RolloutEvalDesc
from tools.utils.decorator import limited_calls
import traceback


class RolloutManager:
    def __init__(self,cfg:DictConfig,agents:Dict[str,Agent]):
        self.cfg=cfg
        self.agents=agents

        self.traning_manager=get_actor("RolloutManager","TrainingManager")
        self.policy_server=get_actor("RolloutManager","PolicyServer")
        self.data_server=get_actor("RolloutManager","DataServer")
        self.monitor=get_actor("RolloutManager","Monitor")

        RolloutWorker=ray.remote(num_cpus=1)(rollout_worker.RolloutWorker)
        self.rollout_workers=[
            RolloutWorker.remote(
                self.default_rollout_worker_id(id),
                self.cfg.seed*self.cfg.num_workers+id,
                self.cfg.worker,
                self.agents
            ) for id in range(self.cfg.num_workers)
        ]

        self.worker_pool=ray.util.ActorPool(self.rollout_workers)

        self.stop_flag=True
        self.stop_flag_lock=threading.Lock()
        # cannot start two rollout tasks
        self.semaphore=threading.Semaphore(value=1)

        self.batch_size=self.cfg.num_workers
        self.data_buffer_max_size=self.batch_size*5

        self.rollout_epoch=0
        self.rollout_epoch_lock=threading.Lock()

    @staticmethod
    def default_rollout_worker_id(id):
        return "rollout_worker_{}".format(id)

    def _rollout_loop(self,rollout_desc):
        self.data_buffer=queue.Queue(maxsize=self.data_buffer_max_size)
        self.data_buffer_lock=threading.Lock()
        self.condition=threading.Condition(self.data_buffer_lock)

        with self.stop_flag_lock:
            assert self.stop_flag
            self.stop_flag=False

        with self.rollout_epoch_lock:
            rollout_epoch=self.rollout_epoch
        for _ in range(self.cfg.num_workers):
            self.worker_pool.submit(
                lambda worker,v: worker.rollout.remote(rollout_desc,rollout_epoch),value=None
            )

        while True:
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            # wait for a rollout to be complete
            result=self.worker_pool.get_next_unordered()
            # start a new task for this available process
            with self.rollout_epoch_lock:
                rollout_epoch=self.rollout_epoch
            self.worker_pool.submit(
                lambda worker,v: worker.rollout.remote(rollout_desc,eval=False,rollout_epoch=rollout_epoch),value=None
            )
            with self.data_buffer_lock:
                self.data_buffer.put_nowait(result)
                while self.data_buffer.qsize()>self.data_buffer_max_size:
                    self.data_buffer.get_nowait()
                if self.data_buffer.qsize()>self.batch_size:
                    self.condition.notify()

                    # FIXME(jh) we have to wait all tasks to terminate? any better way?
        while True:
            if self.worker_pool.has_next():
                self.worker_pool.get_next_unordered()
            else:
                break

        self.data_buffer=None
        self.data_buffer_lock=None
        self.condition=None

    def stop_rollout(self):
        with self.stop_flag_lock:
            self.stop_flag=True

    @limited_calls("semaphore")
    def rollout(self,rollout_desc:RolloutDesc):
        with self.rollout_epoch_lock:
            self.rollout_epoch=0

        self.expr_log_dir=ray.get(self.monitor.get_expr_log_dir.remote())
        self.agent_id=rollout_desc.agent_id
        self.policy_id=rollout_desc.policy_id

        # create table
        table_name=rollout_desc.kwargs["table_name"]
        ray.get(self.data_server.create_table.remote(table_name))

        self._rollout_loop=threading.Thread(target=self._rollout_loop,args=(rollout_desc,))
        self._rollout_loop.start()

        max_rollout_epoch=rollout_desc.kwargs["max_rollout_epoch"]

        # TODO use stopper
        try:
            best_reward=-np.inf
            while True:
                with self.rollout_epoch_lock:
                    if self.rollout_epoch>=max_rollout_epoch:
                        break
                    self.rollout_epoch+=1
                    rollout_epoch=self.rollout_epoch
                results=self.get_batched_results()

                # log to tensorboard, etc...
                log_tasks=[]
                for key,value in results.items():
                    tag="Rollout/{}/{}/{}".format(
                        rollout_desc.agent_id,
                        rollout_desc.policy_id,
                        key
                    )
                    log_task=self.monitor.add_scalar.remote(tag,value,global_step=rollout_epoch)
                    log_tasks.append(log_task)
                ray.get(log_tasks)

                # save model periodically
                if rollout_epoch%self.cfg.save_interval==0:
                    self.save_current_model(f"epoch_{rollout_epoch}")

                # TODO(jh) save best model
                reward=results["reward"]
                if reward>=best_reward:
                    best_reward=reward
                    policy_desc=self.pull_policy(self.agent_id,self.policy_id)
                    best_policy_desc=PolicyDesc(
                        self.agent_id,
                        f"{self.policy_id}.best",
                        policy_desc.policy,
                        version=rollout_epoch
                    )
                    ray.get(self.data_server.push.remote(best_policy_desc))

        except Exception as e:
            Logger.error(traceback.format_exc())
            # save model
            self.save_current_model("last")
            raise e

        # save the last model
        self.save_current_model("last")

        # save the best model
        best_policy_desc=self.pull_policy(self.agent_id,f"{self.policy_id}.best")
        self.save_model(best_policy_desc.policy,self.agent_id,self.policy_id,"best")

        # signal tranining_manager to stop training
        ray.get(self.traning_manager.stop_training.remote())

        # training_manager will stop rollout loop, wait here
        self._rollout_loop.join()

        # remove table
        ray.get(self.data_server.remove_table.remote(table_name))

    def pull_policy(self,agent_id,policy_id):
        if policy_id not in self.agents[agent_id].policy_data:
            policy_desc=ray.get(self.policy_server.pull.remote(agent_id,policy_id,old_version=None))
            self.agents[agent_id][policy_id]=policy_desc
        else:
            old_policy_desc=self.agents[agent_id].policy_data[policy_id]
            policy_desc=ray.get(self.policy_server.pull.remote(agent_id,policy_id,old_version=old_policy_desc.version))
            if policy_desc is not None:
                self.agents[agent_id].policy_data[policy_id]=policy_desc
        return policy_desc

    def save_current_model(self,name):
        self.pull_policy(self.agent_id,self.policy_id)
        policy_desc=self.agents[self.agent_id].policy_data[self.policy_id]
        if policy_desc is not None:
            return self.save_model(policy_desc.policy,self.agent_id,self.policy_id,name)

    def save_model(self,policy,agent_id,policy_id,name):
        dump_dir=os.path.join(
            self.expr_log_dir,
            agent_id,
            policy_id,
            name
        )
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        policy.dump(dump_dir)
        return policy

    @limited_calls("semaphore")
    def rollout_eval(self,rollout_eval_desc:RolloutEvalDesc):
        policy_combinations=rollout_eval_desc.policy_combinations
        num_eval_rollouts=rollout_eval_desc.num_eval_rollouts

        # prepare rollout_desc
        # agent_id & policy_id here is dummy
        rollout_descs = [ RolloutDesc(
            agent_id="agent_0",
            policy_id=policy_combination["agent_0"],
            policy_distributions=policy_combination
        ) for policy_combination in policy_combinations
        ]
        rollout_descs*=num_eval_rollouts

        rollout_results=self.worker_pool.map_unordered(
            lambda worker,rollout_desc: worker.rollout.remote(rollout_desc,eval=True),values=rollout_descs
        )

        # reduce
        results=self.reduce_rollout_eval_results(rollout_results)
        return results

    def get_batched_results(self):
        # retrieve data from data buffer
        while True:
            with self.data_buffer_lock:
                self.condition.wait_for(lambda: self.data_buffer.qsize()>self.batch_size)
                if self.data_buffer.qsize()>self.batch_size:
                    rollout_results=[self.data_buffer.get_nowait() for i in range(self.batch_size)]
                    break

        # reduce
        results=self.reduce_rollout_results(rollout_results)
        return results

    def reduce_rollout_results(self,rollout_results):
        results=defaultdict(list)
        for rollout_result in rollout_results:
            # TODO(jh): policy-wise stats
            # NOTE(jh): now in training, we only care about statistics of the agent is trained
            main_agent_id=rollout_result["main_agent_id"]
            # policy_ids=rollout_result["policy_ids"]
            stats=rollout_result["stats"][main_agent_id]
            for k,v in stats.items():
                results[k].append(v)

        for k,v in results.items():
            results[k]=np.mean(v)

        return results

    def reduce_rollout_eval_results(self,rollout_results):
        # {policy_comb: {agent_id: key: [value]}}
        # policy_comb = ((agent_id, policy_id),)
        results=defaultdict(lambda:defaultdict(lambda: defaultdict(list)))
        for rollout_result in rollout_results:
            policy_ids=rollout_result["policy_ids"]
            stats=rollout_result["stats"]
            policy_comb=tuple([(agent_id,policy_id) for agent_id,policy_id in policy_ids.items()])
            for agent_id,agent_stats in stats.items():
                for key,value in agent_stats.items():
                    results[policy_comb][agent_id][key].append(value)

        for policy_comb,stats in results.items():
            for agent_id,agent_stats in stats.items():
                for key,value in agent_stats.items():
                    agent_stats[key]=np.mean(value)

        return results