import numpy as np
import gym
from gym.spaces import Box,Discrete
import cv2
ENV_LIST = ['FetchPickAndPlace-v1',
            'FetchPush-v1',
            'FetchReach-v1',
            'FetchSlide-v1',
            'HandManipulateBlock-v0',
            'HandManipulateEgg-v0',
            'HandManipulatePen-v0',
            'HandReach-v0']

class ROBOTICS(object):
    def __init__(self,env_name,timelimit):
        if env_name not in ENV_LIST:
            raise NotImplementedError
        self.env_name = env_name
        self.timelimit = timelimit
        self.env = gym.make(env_name)
        

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space.spaces
        
        self.obs_low = self.observation_space['observation'].low 
        self.obs_high = self.observation_space['observation'].high

        self.goal_low = self.observation_space['desired_goal'].low
        self.goal_high = self.observation_space['desired_goal'].high

        self.low = np.array(np.concatenate((self.obs_low,self.goal_low),axis=0))
        self.high = np.array(np.concatenate((self.obs_high,self.goal_high),axis=0))
        self.shape = (self.observation_space['observation'].shape[0]+self.observation_space['desired_goal'].shape[0],)
        self.observation_space = Box(low=self.low,high=self.high,shape=self.shape,dtype=np.float32)
        
        self.score = 0
        self.time = 0
    def seed(self,num):
        self.env.seed(num)
    def render(self):
        rgb = self.env.render("rgb_array")
        return rgb
    def get_rgb(self):
        rgb = self.env.render("rgb_array")
        return rgb
    def reset(self):
        self.score = 0
        self.time = 0
        self.obs_with_goal = self.env.reset()
        self.observation = self.obs_with_goal['observation']
        self.goal = self.obs_with_goal['desired_goal']
        self.state = np.concatenate((self.observation,self.goal),axis=-1)
        return self.state
    def step(self,act):
        act = np.clip(act,self.env.action_space.low,self.env.action_space.high)
        self.obs_with_goal,reward,done,info = self.env.step(act)
        self.observation = self.obs_with_goal['observation']
        self.goal = self.obs_with_goal['desired_goal']
        self.score += reward
        self.time += 1
        done = done or (self.time>self.timelimit)
        reward = reward + 1
        self.state = np.concatenate((self.observation,self.goal),axis=-1)
        return self.state,reward,done,info
        
'''
env = ROBOTICS('FetchPickAndPlace-v1',1000)
for i in range(0,200):
    state = env.reset()
    done = False
    while not done:
        rgb = env.render()
        cv2.imshow("rgb",rgb)
        cv2.waitKey(5)
        act = np.random.randn(env.action_space.shape[0])
        next_state,reward,done,_ = env.step(act)
        print(reward)'''
