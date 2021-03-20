import numpy as np
import threading
class Running_Estimator(object):
    def __init__(self,shape,lock,clipvalue=5):
        self._lock = lock
        self._clipvalue = clipvalue 
        self._n = 0
        self._mean = np.zeros(shape) 
        self._var = np.zeros(shape)
    def store(self,x):
        if self._lock.acquire():
            x = np.asarray(x)
            assert x.shape == self._mean.shape
            self._n += 1
            self._oldmean = np.array(self._mean)
            self._mean = self._oldmean + (x-self._mean)/self._n
            self._var = self._var + (x - self._oldmean)*(x - self._mean)
            self._lock.release()
    def normalize(self,x):
        return np.clip((x-self.mean)/(self.std+1e-5),-self._clipvalue,self._clipvalue)

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
        return np.sqrt(self.var/self.n)
    @property
    def shape(self):
        return self._mean.shape

## Test Running_Estimator
'''
data = np.random.rand(1,10)
lock = threading.Condition()
rms = Running_Estimator(data.shape[1],lock)
for i in range(0,data.shape[0]):
    rms.store(data[i])
print("rms mean={}".format(rms.mean))
print("rms std={}".format(rms.std))
print("mean={}".format(np.mean(data,axis=0)))
print("std={}".format(np.std(data,axis=0)))
'''






    



