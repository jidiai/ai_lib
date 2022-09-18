import numpy as np

class SimpleMetric:
    def __init__(self,init_value=0):
        self.value=init_value
        self.ctr=0
    
    def update(self,value):
        self.value+=value
        self.ctr+=1
    
    @property
    def mean(self):
        return self.value/self.ctr if self.ctr>0 else None
    
    
class WindowMetric:
    def __init__(self,window_size):
        self.window_size=window_size
        self.window=[]
        self.ptr=0
        
    def update(self,value):
        if len(self.data)<self.window_size:
            self.window.append(value)
        else:
            self.window[self.ptr]=value
        self.ptr=(self.ptr+1)%self.window_size
        
    @property
    def mean(self):
        return np.mean(self.values) if len(self.data)>0 else None