from typing import Dict
from .table import Table
from tools.utils.logger import Logger
import threading
    
class DataServer:
    def __init__(self,cfg): 
        self.cfg=cfg        
        
        self.tables:Dict[str,Table]={}
        self.table_names=[]
        self.table_lock=threading.Lock()
        
        self.read_timeout = self.cfg.read_timeout
        self.sample_max_usage = self.cfg.sample_max_usage        
        self.episode_capacity = self.cfg.episode_capacity
        self.sampler_type = self.cfg.sampler_type
        
        Logger.warning("Data Server uses {} sampler".format(self.sampler_type))
        
    @staticmethod
    def default_table_name(agent_id,policy_id):
        return "{}_{}".format(agent_id,policy_id)
    
    def create_table(self,table_name): 
        with self.table_lock:
            self.tables[table_name]=Table(capacity=self.episode_capacity,sample_max_usage=self.sample_max_usage,sampler_type=self.sampler_type)
            self.table_names.append(table_name)
            Logger.info("created data table: {}".format(table_name))
        
    def remove_table(self,table_name):
        with self.table_lock:
            if table_name in self.tables:
                self.table_names.remove(table_name)
                self.tables.pop(table_name)
            Logger.info("removed data table: {}".format(table_name))
            
    def get_table_stats(self,table_name):
        try:
            with self.table_lock:
                statistics=self.tables[table_name].get_statistics()
            return statistics
        except KeyError:
            info = "table {} is not found".format(table_name)
            Logger.warning(info)
            return {}
    
    def save_data(self, table_name, data):
        try:
            with self.table_lock:
                table:Table = self.tables[table_name]
            table.write(data)
        except Exception:
            Logger.warning("table {} is not found".format(table_name))
    
    def sample_data(self, table_name, batch_size, wait=False):
        try:
            with self.table_lock:
                table:Table = self.tables[table_name]
            samples = None
            samples = table.read(batch_size,timeout=self.timeout)
            if samples is not None:
                return samples,True
            else:
                return samples,False
        except Exception:
            samples=None
            Logger.warning("table {} is not found".format(table_name))
            return samples,False

