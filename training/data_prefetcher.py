import queue
import threading
import time
from turtle import settiltangle
import ray
import tree
from tools.desc.task_desc import PrefetchingDesc
from tools.utils.logger import Logger
from tools.utils.timer import Timer,global_timer
from tools.utils.typing import BufferDescription
from tools.utils.typing import (
    BufferDescription,
    PolicyID,
    AgentID,
    Dict,
    List,
    Any,
    Union,
    Tuple,
    Status,
)
import numpy as np
import settings
import torch
from tools.utils.remote import get_actor
from tools.utils.decorator import limited_calls

class DataPrefetcher:
    def __init__(self, consumers):
        self.consumers=consumers
        self.data_server=get_actor("DataFetcher","DataServer")

        self.timer=global_timer
        
        self.stop_flag=True
        self.stop_flag_lock=threading.Lock()
        # cannot start two rollout tasks
        self.semaphore=threading.Semaphore(value=1)
    
    @limited_calls("semaphore")
    def prefetch(self,prefetching_desc:PrefetchingDesc):
        with self.stop_flag_lock:
            assert self.stop_flag
            self.stop_flag=False
        
        while True:
            with self.stop_flag_lock:
                if self.stop_flag:
                    break
            self.request_data(prefetching_desc)
        Logger.warning("DataFetcher main_task() ends")
        
    def stop_prefetching(self):
        with self.stop_flag_lock:
            self.stop_flag=True
        
    def request_data(self, prefetching_desc:PrefetchingDesc):
        self.timer.record("sample_from_remote_start")
        data, ok = ray.get(
            self.data_server.sample_data.remote(prefetching_desc.table_name,prefetching_desc.batch_size)
        )
        if not ok:
            return
        else:
            assert data is not None
        
        self.timer.time("sample_from_remote_start","sample_from_remote_end","sample_from_remote")        
        
        base_num=int(len(data)/len(self.consumers))
        nums=np.full(len(self.consumers),fill_value=base_num)
        remainder=len(data)%len(self.consumers)
        nums[:remainder]+=1
        
        indices=np.cumsum(nums)[:-1]
        
        samples_list=np.split(data,indices)        
        
        tasks=[]
        for consumer,samples in zip(self.consumers,samples_list):
            samples=self.concat(samples)
            task=consumer.local_queue_put.remote(samples)
            tasks.append(task)
        ray.get(tasks)
            
        
    def concat(self,samples):        
        ret={}
        for k,v in samples[0].items():
            # recursively stack
            if isinstance(v,dict):
                ret[k]=self.concat([sample[k] for sample in samples])
            elif isinstance(v,np.ndarray):
                ret[k]=np.concatenate([sample[k] for sample in samples])
            else:
                pass
        return ret
    
# TODO(jh): ?useful or not.
class GPUPreLoadQueueWrapper:
    '''
    Modified from https://docs.ray.io/en/latest/_modules/ray/train/torch.html#prepare_data_loader
    '''
    def __init__(
        self, queue:queue.Queue, device: torch.device=torch.device("cuda"), auto_transfer: bool=True
    ):

        self._queue=queue
        self.device = device
        # disable auto transfer (host->device) if cpu is used
        self._auto_transfer = auto_transfer if device.type == "cuda" else False
        # create a new CUDA stream to move data from host to device concurrently
        self._memcpy_stream = (
            torch.cuda.Stream()
            if device.type == "cuda" and self._auto_transfer
            else None
        )
        self.next_batch = None
        self._prefetch_next_batch(block=True)

    def _move_to_device(self, data):
        if data is None:
            return None

        def to_device(data):
            if isinstance(data,dict):
                ret={}
                for k,v in data.items():
                    ret[k]=to_device(v)
                return ret
            elif isinstance(data,list):
                ret=[]
                for v in data:
                    ret.append(to_device(v))
                return ret
            elif isinstance(data,np.ndarray):
                data=torch.from_numpy(data)
                ret=data.to(self.device,non_blocking=self._auto_transfer)
                return ret
            elif isinstance(data,torch.Tensor):
                ret=data.to(self.device,non_blocking=self._auto_transfer)
                return ret
            return ret

        with torch.cuda.stream(self._memcpy_stream):
            return to_device(data)

    def _wait_for_batch(self, item):
        if self._memcpy_stream is None:
            return
        # Reference:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
        # The training stream (current) needs to wait until
        # the memory copy stream finishes.
        curr_stream = torch.cuda.current_stream()
        curr_stream.wait_stream(self._memcpy_stream)
        # When a tensor is used by CUDA streams different from
        # its original allocator, we need to call ``record_stream``
        # to inform the allocator of all these streams. Otherwise,
        # the tensor might be freed once it is no longer used by
        # the creator stream.
        for i in item:
            # The Pytorch DataLoader has no restrictions on what is outputted for
            # each batch. We should only ``record_stream`` if the item has the
            # ability to do so.
            try:
                i.record_stream(curr_stream)
            except AttributeError:
                pass

    def _prefetch_next_batch(self,block):
        next_batch = self._queue.get(block)
        self.next_batch = self._move_to_device(next_batch)

    def get(self,block=True):
        next_batch = self.next_batch
        self._wait_for_batch(next_batch)
        self._prefetch_next_batch(block)
        return next_batch
    
    def put(self,data,block=True,timeout=None):
        self._queue.put(data,block,timeout)
        
    @staticmethod
    def to_pin_memory(data):
        if isinstance(data,dict):
            ret={}
            for k,v in data.items():
                ret[k]=GPUPreLoadQueueWrapper.to_pin_memory(v)
            return ret
        elif isinstance(data,list):
            ret=[]
            for v in data:
                ret.append(GPUPreLoadQueueWrapper.to_pin_memory(v))
            return ret
        elif isinstance(data,np.ndarray):
            data=torch.from_numpy(data)
            ret=data.pin_memory()
            return ret
        elif isinstance(data,torch.Tensor):
            ret=data.pin_memory()
            return ret
        return ret