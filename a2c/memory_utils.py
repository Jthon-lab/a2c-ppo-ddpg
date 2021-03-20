import numpy as np
import network_utils as net_utils

# gamma and lam are the hyper-parameters for GAE 
# Local Memory is a private memory for each worker 
class Local_Memory(object):
    def __init__(self,ob_space,ac_space,size,gamma=0.99,lam=0.95):
        size = int(size)
        self.obs_buffer = np.zeros(net_utils.combined_shape(size,ob_space.shape),dtype=np.float32)
        self.act_buffer = np.zeros(net_utils.combined_shape(size,ac_space.shape),dtype=np.float32)
        self.adv_buffer = np.zeros(size,dtype=np.float32)
        self.rew_buffer = np.zeros(size,dtype=np.float32)
        self.ret_buffer = np.zeros(size,dtype=np.float32)

        self.gamma,self.lam = gamma,lam
        self.ptr,self.path_start_idx,self.max_size = 0,0,size
    
    def store(self,obs,act,rew,val):
        assert self.ptr < self.max_size
        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.rew_buffer[self.ptr] = rew
        self.ret_buffer[self.ptr] = val
        self.ptr += 1
    
    def finish_path(self,last_val = 0):
        path_slice = slice(self.path_start_idx,self.ptr)
        rews = np.append(self.rew_buffer[path_slice],last_val)
        vals = np.append(self.ret_buffer[path_slice],last_val)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buffer[path_slice] = net_utils.discount_cumsum(deltas,self.gamma*self.lam)
        self.ret_buffer[path_slice] = net_utils.discount_cumsum(rews,self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def full(self):
        return self.ptr == self.max_size
    
    def get(self):
        assert self.ptr == self.max_size
        self.ptr,self.path_start_idx = 0,0
        return self.obs_buffer,self.act_buffer,self.adv_buffer,self.ret_buffer

## Global_Memory is a shared memory among all workers
## After every worker finished the collecting the data, we use the global memory for synchorinize
## And use the shared memory for model updating
class Global_Memory(object):
    def __init__(self,ob_space,ac_space,parallels=8,train_iters=1):
        self.parallels = parallels
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.collect_thread = 0

        self.nminibatch = train_iters
        self.batch_index = 0

        self.obs_buffer = np.zeros(net_utils.combined_shape(0,shape=ob_space.shape),dtype=np.float32)
        self.act_buffer = np.zeros(net_utils.combined_shape(0,shape=ac_space.shape),dtype=np.float32)
        self.adv_buffer = np.zeros((0,),dtype=np.float32)
        self.ret_buffer = np.zeros((0,),dtype=np.float32)
    
    def collect(self,thread_obs,thread_act,thread_adv,thread_ret):
        self.collect_thread += 1
        self.obs_buffer = np.concatenate((self.obs_buffer,thread_obs),axis=0)
        self.act_buffer = np.concatenate((self.act_buffer,thread_act),axis=0)
        self.adv_buffer = np.concatenate((self.adv_buffer,thread_adv),axis=0)
        self.ret_buffer = np.concatenate((self.ret_buffer,thread_ret),axis=0)

        shuffle_index = np.arange(self.obs_buffer.shape[0])
        np.random.shuffle(shuffle_index)

        self.obs_buffer = self.obs_buffer[shuffle_index]
        self.act_buffer = self.act_buffer[shuffle_index]
        self.adv_buffer = self.adv_buffer[shuffle_index]
        self.ret_buffer = self.ret_buffer[shuffle_index]
        
    def get(self):
        assert self.full() == True
        size = self.obs_buffer.shape[0]
        batch_size = int(size//self.nminibatch)
        norm_adv_buf = (self.adv_buffer - np.mean(self.adv_buffer))/(np.std(self.adv_buffer)+1e-5)
        start_index = int(self.batch_index * batch_size)
        end_index = int(np.clip((self.batch_index + 1)*batch_size,0,size))

        obs_batch = self.obs_buffer[start_index:end_index]
        act_batch = self.act_buffer[start_index:end_index]
        adv_batch = norm_adv_buf[start_index:end_index]
        ret_batch = self.ret_buffer[start_index:end_index]
        
        self.batch_index = (self.batch_index + 1)%self.nminibatch
        return obs_batch,act_batch,adv_batch,ret_batch

    def full(self):
        assert self.collect_thread <= self.parallels
        return self.collect_thread >= self.parallels
        
    def clear(self):
        self.obs_buffer = np.zeros(net_utils.combined_shape(0,shape=self.ob_space.shape),dtype=np.float32)
        self.act_buffer = np.zeros(net_utils.combined_shape(0,shape=self.ac_space.shape),dtype=np.float32)
        self.adv_buffer = np.zeros((0,),dtype=np.float32)
        self.ret_buffer = np.zeros((0,),dtype=np.float32)
        self.collect_thread = 0

