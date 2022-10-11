class RateLimiter:
    def __init__(self,table,min_size=1,r_w_ratio=None):
        self.table=table
        self.min_size=min_size
        self.r_w_ratio=r_w_ratio
    
    def is_reading_available(self,batch_size):
        if self.table.write_num<self.min_size:
            return False
        if self.r_w_ratio is not None:
            max_read_num=self.table.write_num*self.r_w_ratio
            if self.table.read_num+batch_size>=max_read_num:
                return False
        # else
        return True
