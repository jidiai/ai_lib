import queue
import threading
import traceback
from typing import List
import ray
from utils.desc.task_desc import PrefetchingDesc
from utils.logger import Logger
from utils.timer import Timer, global_timer
import numpy as np
import torch
from utils.decorator import limited_calls


class DataPrefetcher:
    def __init__(self, cfg, consumers, data_servers):
        self.cfg = cfg
        self.consumers = consumers
        self.data_servers = data_servers

        self.timer = global_timer

        self.stop_flag = True
        self.stop_flag_lock = threading.Lock()
        # cannot start two rollout tasks
        self.semaphore = threading.Semaphore(value=1)
        Logger.info("DataPrefetcher initialized")

    @limited_calls("semaphore")
    def prefetch(self, prefetching_descs: List[PrefetchingDesc]):
        try:
            assert len(prefetching_descs) == len(self.data_servers)
            with self.stop_flag_lock:
                assert self.stop_flag
                self.stop_flag = False

            while True:
                with self.stop_flag_lock:
                    if self.stop_flag:
                        break
                self.request_data(prefetching_descs)
            Logger.warning("DataFetcher main_task() ends")
        except:
            print(traceback.format_exc())

    def stop_prefetching(self):
        with self.stop_flag_lock:
            self.stop_flag = True

    def request_data(self, prefetching_descs: List[PrefetchingDesc]):
        self.timer.record("sample_from_remote_start")

        data_list = []
        for data_server, prefetching_desc in zip(self.data_servers, prefetching_descs):
            data, ok = ray.get(
                data_server.sample.remote(prefetching_desc.table_name, prefetching_desc.batch_size)
            )
            if not ok:
                return
            else:
                assert isinstance(data, list) and isinstance(data[0],
                                                             dict), "type of data: {}, type of data[0]: {}".format(
                    type(data), type(data[0]))
                data_list.append(data)

        # merge data
        samples = []
        for i in range(len(data_list[0])):
            sample = {}
            for data in data_list:
                sample.update(data[i])
            samples.append(sample)

        self.timer.time("sample_from_remote_start", "sample_from_remote_end", "sample_from_remote")

        base_num = int(len(samples) / len(self.consumers))
        nums = np.full(len(self.consumers), fill_value=base_num)
        remainder = len(samples) % len(self.consumers)
        nums[:remainder] += 1

        indices = np.cumsum(nums)[:-1]

        samples_list = np.split(samples, indices)

        tasks = []
        for consumer, samples in zip(self.consumers, samples_list):
            samples = self.stack(samples)
            task = consumer.local_queue_put.remote(samples)
            tasks.append(task)
        ray.get(tasks)

    def stack(self, samples):
        ret = {}
        for k, v in samples[0].items():
            # recursively stack
            if isinstance(v, dict):
                ret[k] = self.stack([sample[k] for sample in samples])
            elif isinstance(v, np.ndarray):
                ret[k] = np.stack([sample[k] for sample in samples])
            elif isinstance(v, list):
                ret[k] = [self.stack([sample[k][i] for sample in samples]) for i in range(len(v))]
            else:
                raise NotImplementedError(f'v = {v}, k={k}, ret = {[sample[k] for sample in samples]}')
        return ret

    def concat(self, samples):
        ret = {}
        for k, v in samples[0].items():
            # recursively stack
            if isinstance(v, dict):
                ret[k] = self.concat([sample[k] for sample in samples])
            elif isinstance(v, np.ndarray):
                ret[k] = np.concatenate([sample[k] for sample in samples])
            elif isinstance(v, list):
                ret[k] = [self.concat([sample[k][i] for sample in samples]) for i in range(len(v))]
            else:
                raise NotImplementedError
        return ret


# TODO(jh): ?useful or not.
class GPUPreLoadQueueWrapper:
    '''
    Modified from https://docs.ray.io/en/latest/_modules/ray/train/torch.html#prepare_data_loader
    '''

    def __init__(
            self, queue: queue.Queue, device: torch.device = torch.device("cuda"), auto_transfer: bool = True
    ):

        self._queue = queue
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
        # self._prefetch_next_batch(block=True)

    def _move_to_device(self, data):
        if data is None:
            return None

        def to_device(data):
            if isinstance(data, dict):
                ret = {}
                for k, v in data.items():
                    ret[k] = to_device(v)
                return ret
            elif isinstance(data, list):
                ret = []
                for v in data:
                    ret.append(to_device(v))
                return ret
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
                ret = data.to(self.device, non_blocking=self._auto_transfer)
                return ret
            elif isinstance(data, torch.Tensor):
                ret = data.to(self.device, non_blocking=self._auto_transfer)
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

    def _prefetch_next_batch(self, block):
        next_batch = self._queue.get(block)
        self.next_batch = self._move_to_device(next_batch)

    def get(self, block=True):
        next_batch = self.next_batch
        self._wait_for_batch(next_batch)
        self._prefetch_next_batch(block)
        return next_batch

    def put(self, data, block=True, timeout=None):
        self._queue.put(data, block, timeout)

    @staticmethod
    def to_pin_memory(data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = GPUPreLoadQueueWrapper.to_pin_memory(v)
            return ret
        elif isinstance(data, list):
            ret = []
            for v in data:
                ret.append(GPUPreLoadQueueWrapper.to_pin_memory(v))
            return ret
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            ret = data.pin_memory()
            return ret
        elif isinstance(data, torch.Tensor):
            ret = data.pin_memory()
            return ret
        return ret