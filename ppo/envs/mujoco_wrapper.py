import numpy as np
import gym
from gym.spaces import Box,Discrete
import cv2

ENV_LIST=['Ant-v3','HalfCheetah-v3','Hopper-v3','Humanoid-v3','HumanoidStandup-v3','InvertedDoublePendulum-v3',
          'InvertedPendulum-v3','Reacher-v3','Swimmer-v3','Walker2d-v3']
class MUJOCO(object):
    def __init__(self,env_name,env_seed,timelimit):
        if env_name not in ENV_LIST:
            raise NotImplementedError
        self.env_name = env_name
        self.env_seed = env_seed
        self.timelimit = timelimit
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.score = 0
        self.time = 0
    def seed(self,num):
        self.env.seed(num)
    def render(self):
        self.env.render("human")
    def get_rgb(self):
        self.rgb = self.env.render("rgb_array")
        return self.rgb
    def reset(self):
        self.score = 0 
        self.time = 0
        self.state = self.env.reset()
        self.env.seed(self.env_seed)
        return np.array(self.state)
    def step(self,act):
        #clip the action
        act = np.clip(act,self.env.action_space.low,self.env.action_space.high)
        self.state,reward,done,info = self.env.step(act)
        self.score += reward
        self.time += 1
        done = done or (self.time>self.timelimit)
        return np.array(self.state),reward,done,info

'''
env = MUJOCO("Humanoid-v2",1000)
for i in range(0,200):
    state = env.reset()
    done = False
    while not done:
        env.render()
        act = np.random.randn(env.action_space.shape[0])
        next_state,reward,done,_ = env.step(act)
        print(reward)'''
    
