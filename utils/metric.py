from collections import deque
import numpy as np

class SimpleMetric:
    def __init__(self,**kwargs):
        init_value=kwargs.get("init_value",0)
        self.value=init_value
        self.ctr=0
    
    def update(self,value):
        self.value+=value
        self.ctr+=1
    
    def mean(self):
        return self.value/self.ctr if self.ctr>0 else None
    
    
class SlidingMetric:
    def __init__(self,**kwargs):
        window_size=kwargs["window_size"]
        init_list=kwargs.get("init_list",[])
        self.window_size=window_size
        self.window=list(init_list)
        self.ptr=len(self.window)
        
    def update(self,value):
        if len(self.window)<self.window_size:
            self.window.append(value)
        else:
            self.window[self.ptr]=value
        self.ptr=(self.ptr+1)%self.window_size
        
    def mean(self):
        return np.mean(self.window) if len(self.window)>0 else None
    
class Metrics:
    def __init__(self,metric_cfgs):
        self.metric_cfgs=metric_cfgs
        self.metrics={}
        for metric_name,metric_cfg in metric_cfgs.items():
            if metric_cfg["type"]=="simple":
                self.metrics[metric_name]=SimpleMetric(**metric_cfg)
            elif metric_cfg["type"]=="sliding":
                self.metrics[metric_name]=SlidingMetric(**metric_cfg)
            else:
                raise NotImplementedError
            
    def update(self,results):
        for k,v in results.items():
            if k in self.metrics:
                self.metrics[k].update(v)
    
    def get_means(self,metric_names=None):
        if metric_names is None:
            metric_names=list(self.metrics.keys())
        metric_means={}
        for metric_name in metric_names:
            metric_means[metric_name]=self.metrics[metric_name].mean()
        return metric_means
            