import numpy as np
import network_utils as net_utils

# gamma and lam are the hyper-parameters for GAE 
# Local Memory is a private memory for each worker 
class Memory_Buffer(object):
    def __init__(self,thread_num,ob_space,ac_space,lock,size=10000,kstep=1,batch_size=64):
        self.thread_num = thread_num
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.size = size
        self.lock = lock
        self.batch_size = batch_size

        self.s1_buffer = np.zeros(net_utils.combined_shape(size,ob_space.shape),dtype=np.float32)
        self.s2_buffer = np.zeros(net_utils.combined_shape(size,ob_space.shape),dtype=np.float32)
        self.act_buffer = np.zeros(net_utils.combined_shape(size,ac_space.shape),dtype=np.float32)
        self.rews_buffer = np.zeros(net_utils.combined_shape(size,kstep+1),dtype=np.float32)
        self.dones_buffer = np.zeros(net_utils.combined_shape(size,(1,)),dtype=np.int32)

        self.current_thread = 0
        self.ptr = 0
        self.current_size = 0
    def store(self,s1,s2,act,rews,done):
        if self.lock.acquire():
            self.s1_buffer[self.ptr] = s1
            self.s2_buffer[self.ptr] = s2
            self.act_buffer[self.ptr] = act
            self.rews_buffer[self.ptr] = rews
            self.dones_buffer[self.ptr] = done     
        
            self.ptr = (self.ptr+1)%self.size
            self.current_size = min(self.current_size+1,self.size)
            self.lock.release()

    def get_batch(self):
        random_indexes = np.random.choice(self.current_size,self.batch_size)
        s1_batch = self.s1_buffer[random_indexes]
        s2_batch = self.s2_buffer[random_indexes]
        act_batch = self.act_buffer[random_indexes]
        rews_batch = self.rews_buffer[random_indexes]
        dones_batch = self.dones_buffer[random_indexes]
        return s1_batch,s2_batch,act_batch,rews_batch,dones_batch
    
    def collect_thread(self):
        self.current_thread += 1
    def full(self):
        return self.current_thread >= self.thread_num
        
    def clear_thread(self):
        self.current_thread = 0




