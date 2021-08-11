import numpy as np


class Replay_buffer(object):
    def __init__(self, max_size, trajectory_property):
        self.storage = []
        self.max_size = max_size

        self.property_list = ['states', 'states_next', 'rewards', 'dones']
        self.property_additional = trajectory_property
        self.properties_all = self.property_list + self.property_additional
        self.item_buffers = dict()
        self.step_index_by_env = 0

        self.buffer_dict = dict()
        self.buffer_dict_clear()
        self.ptr = 0

    def buffer_dict_clear(self):
        for item in self.properties_all:
            self.buffer_dict[item] = list()

    def init_item_buffers(self):
        for p in self.properties_all:
            self.item_buffers[p] = ItemBuffer(self.max_size, p)

    def insert(self, item_name:str, agent_id:int, data:np.ndarray, step=None):
        if item_name == 'dones':
            agent_id = 0
        self.item_buffers[item_name].insert(agent_id, step, data)

    def sample(self, batch_size):
        self.buffer_dict_clear()
        data_length = len(self.item_buffers["action"].data)
        ind = np.random.randint(0, data_length, size=batch_size)
        for name, item_buffer in self.item_buffers.items():
            for i in ind:
                self.buffer_dict[name].append(np.array(item_buffer.data[i], copy=False))
        return self.buffer_dict

    def get_trajectory(self):
        self.buffer_dict_clear()
        data_length = len(self.item_buffers["action"].data)
        for name, item_buffer in self.item_buffers.items():
            for i in range(data_length):
                self.buffer_dict[name].append(np.array(item_buffer.data[i], copy=False))
        return self.buffer_dict

    def get_step_data(self):
        self.buffer_dict_clear()
        for name, item_buffer in self.item_buffers.items():
            self.buffer_dict[name] = item_buffer.data[0]
        return self.buffer_dict

    def item_buffer_clear(self):
        for p in self.properties_all:
            self.item_buffers[p].clear()


class ItemBuffer(object):
    def __init__(self, max_size, name):
        self.name = name
        self.max_size = max_size
        self.A = 1
        self.data = list()
        self.ptr = 0

    def insert(self, agent_id:int, step:int, data:np.ndarray):
        if len(self.data) == self.max_size:
            self.data.pop(0)
        self.data.append(data)

    def clear(self):
        del self.data[:]