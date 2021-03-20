import numpy as np
import matplotlib.pyplot as plt
## Reference from openai-baselines 
## https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class ActionNoise(object):
    def reset(self):
        pass

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def __call__(self):
        return np.random.normal(self.mu, self.sigma)
    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self,mu,sigma,theta=.15,dt=1e-2,x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return np.clip(x,-1,1)
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Test OU noise
'''dim = 5
mu = np.zeros((dim,),np.float32)
sigma = np.ones((dim,),np.float32)*0.1
noise = OrnsteinUhlenbeckActionNoise(mu,sigma)
figure = plt.figure()
record_noise = np.empty((0,dim),np.float32)
for i in range(0,500):
    print(record_noise.shape)
    print(noise().shape)
    record_noise = np.concatenate((record_noise,[noise()]),axis=0)
for i in range(record_noise.shape[1]):
    plt.plot(record_noise[:,i])
plt.show()'''


    
