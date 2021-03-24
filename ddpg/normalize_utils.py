import numpy as np
import threading
from network_utils import discount_cumsum
class Running_Estimator(object):
    def __init__(self,shape,lock,clipvalue=5):
        self._lock = lock
        self._clipvalue = clipvalue 
        self._n = 0
        self._mean = np.zeros(shape) 
        self._var = np.zeros(shape)
    def normalize(self,x):
        if self._lock.acquire():
            x = np.asarray(x)
            assert x.shape == self._mean.shape
            self._n += 1
            self._oldmean = np.array(self._mean)
            self._mean = self._oldmean + (x-self._mean)/self._n
            self._var = self._var + (x - self._oldmean)*(x - self._mean)
            self._lock.release()
        return np.clip((x-self.mean)/(self.std+1e-5),-self._clipvalue,self._clipvalue)
    def normalize_without_mean(self,x):
        if self._lock.acquire():
            x = np.asarray(x)
            assert x.shape == self._mean.shape
            self._n += 1
            self._oldmean = np.array(self._mean)
            self._mean = self._oldmean + (x-self._mean)/self._n
            self._var = self._var + (x - self._oldmean)*(x - self._mean)
            self._lock.release()
        return np.clip((x)/(self.std+1e-5),-self._clipvalue,self._clipvalue)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._mean
    @property
    def var(self):
        return self._var if self._n > 1 else np.square(self._mean)
    @property
    def std(self):
        if self.n <= 1:
            return np.sqrt(np.abs(self._mean))
        return np.sqrt(self.var/self.n)
    @property
    def shape(self):
        return self._mean.shape

class Running_Reward_Normalizer(object):
    def __init__(self,shape,lock,clipvalue=5):
        self._lock = lock
        self._clipvalue = clipvalue 
        self._n = 0
        self._mean = np.zeros(shape) 
        self._var = np.zeros(shape)
    def store(self,rewards,gamma):
        if self._lock.acquire():
            x = discount_cumsum(rewards,gamma)[0]
            x = np.asarray(x)
            assert x.shape == self._mean.shape
            self._n += 1
            self._oldmean = np.array(self._mean)
            self._mean = self._oldmean + (x-self._mean)/self._n
            self._var = self._var + (x - self._oldmean)*(x - self._mean)
            self._lock.release()
    def normalize(self,x):
        if self.n < 1:
            return x
        return np.clip((x-self.mean)/(np.clip(self.std,0,100)+1e-5),-self._clipvalue,self._clipvalue)
    def normalize_without_mean(self,x):
        if self.n < 1:
            return x
        return np.clip((x)/(np.clip(self.std,0,100)+1e-5),-self._clipvalue,self._clipvalue)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._mean
    @property
    def var(self):
        return self._var if self._n > 1 else np.square(self._mean)
    @property
    def std(self):
        if self.n < 1:
            return np.sqrt(np.abs(self._mean))
        return np.sqrt(self.var/self.n)
    @property
    def shape(self):
        return self._mean.shape





    



