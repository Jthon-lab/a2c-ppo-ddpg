import numpy as np
import gym
from gym.spaces import Box,Discrete
import cv2

TOY_D_LIST=["CartPole-v0","MountainCar-v0","LunarLander-v2"]
TOY_C_LIST=["Acrobot-v1","MountainCarContinuous-v0","Pendulum-v0","LunarLanderContinuous-v2"]
class TOY_Discrete(object):
    def __init__(self,env_name,env_seed,timelimit):
        if env_name not in TOY_D_LIST:
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
        rgb = self.env.render("rgb_array")
        return rgb
    def get_rgb(self):
        rgb = self.env.render("rgb_array")
        return rgb
    def reset(self):
        self.score = 0
        self.time = 0
        self.state = self.env.reset()
        self.env.seed(self.env_seed)
        return np.array(self.state)
    def step(self,act):
        self.state,reward,done,info = self.env.step(act)
        self.score += reward
        self.time += 1
        done = done or (self.time>self.timelimit)
        return np.array(self.state),reward,done,info

class TOY_Continuous(object):
    def __init__(self,env_name,env_seed,timelimit):
        if env_name not in TOY_C_LIST:
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
        rgb = self.env.render("rgb_array")
        return rgb
    def get_rgb(self):
        rgb = self.env.render("rgb_array")
        return rgb
    def reset(self):
        self.score = 0
        self.time = 0
        self.state = self.env.reset()
        self.env.seed(self.env_seed)
        return np.array(self.state)
    def step(self,act):
        self.state,reward,done,info = self.env.step(act)
        self.score += reward
        self.time += 1
        done = done or (self.time>self.timelimit)
        return np.array(self.state),reward,done,info

