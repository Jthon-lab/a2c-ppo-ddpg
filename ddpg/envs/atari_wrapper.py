import cv2
import numpy as np
import gym
from gym.spaces import Box,Discrete
import numpy as np
import copy
class ATARI(object):
    def __init__(self,env_name,timelimit,stack_frame=1,action_repeat=4,color=True,image_size=128):
        self.env_name = env_name
        self.timelimit = timelimit
        self.time = 0
        self.score = 0

        self.color = color
        self.stack_frame = stack_frame
        self.action_repeat = action_repeat
        self.image_size =image_size

        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space
        if color==True:
            self.observation_space = Box(-1,1,(image_size,image_size,3*stack_frame,))
            self.channel = 3
        else:
            self.observation_space = Box(-1,1,(image_size,image_size,1*stack_frame,))
            self.channel = 1
        self.state = np.zeros(self.observation_space.shape,dtype=np.float32)
    def __normalize__(self,image):
        image = np.array(image/127.5-1,np.float32)
        return image
    def __denormalize__(self,image):
        image = np.array((image+1)*127.5,np.uint8)
        return image
    def seed(self,seed_num):
        self.env.seed(seed_num)
        
    def reset(self):
        self.score = 0
        obs = self.env.reset()
        obs = cv2.cvtColor(obs,cv2.COLOR_BGR2RGB)
        self.rgb = np.array(obs)
        obs = cv2.resize(obs,(self.image_size,self.image_size))
        if self.color == False:
            obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
            obs = np.expand_dims(obs,-1)
        obs = self.__normalize__(obs)

        self.state[:,:,0:self.channel] = obs
        self.time = 0
        self.lives = copy.deepcopy(self.env.unwrapped.ale.lives())
        return np.array(self.state)

    def render(self):
        return self.rgb
    def get_rgb(self):
        rgb = cv2.cvtColor(self.rgb,cv2.COLOR_BGR2RGB)
        return rgb
    def step(self,act):
        assert act>=0 and act<self.action_space.n
        repeat_reward = 0
        for i in range(0,self.action_repeat):
            next_obs,reward,done,info = self.env.step(act)
            repeat_reward += reward
            if done==True:
                break
        next_obs = cv2.cvtColor(next_obs,cv2.COLOR_BGR2RGB)
        self.rgb = np.array(next_obs)
        next_obs = cv2.resize(next_obs,(self.image_size,self.image_size))
        if self.color == False:
            next_obs = cv2.cvtColor(next_obs,cv2.COLOR_BGR2GRAY)
            next_obs = np.expand_dims(next_obs,-1)
        next_obs = self.__normalize__(next_obs)
        for i in range(1,self.stack_frame):
            self.state[:,:,self.channel*(self.stack_frame-i):self.channel*(self.stack_frame-i+1)] = self.state[:,:,self.channel*(self.stack_frame-i-1):self.channel*(self.stack_frame-i)]
        self.state[:,:,0:self.channel] = next_obs
        self.time+=1
        done = done or (self.time>=self.timelimit)
        if self.time>self.timelimit:
            print("Warning: Exceed the permitted time limit")
        self.score += repeat_reward

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives>0:
            done = True
        return np.array(self.state),repeat_reward,done,info
'''
#Test Wrapper
env = ATARI("Breakout-v0",500,4,1,True,128)
state = env.reset()
done = False
while not done:
    next_state,reward,done,info = env.step(np.random.choice(env.action_space.n))
    while True:
        if cv2.waitKey(5)==27:
            break
        cv2.imshow("rgb",env.render())
        cv2.imshow("state1",np.array((state[:,:,0:3]+1)*127.5,np.uint8))
        cv2.imshow("state2",np.array((state[:,:,3:6]+1)*127.5,np.uint8))
        cv2.imshow("state3",np.array((state[:,:,6:9]+1)*127.5,np.uint8))
        cv2.imshow("state4",np.array((state[:,:,9:12]+1)*127.5,np.uint8))

        cv2.imshow("nstate1",np.array((next_state[:,:,0:3]+1)*127.5,np.uint8))
        cv2.imshow("nstate2",np.array((next_state[:,:,3:6]+1)*127.5,np.uint8))
        cv2.imshow("nstate3",np.array((next_state[:,:,6:9]+1)*127.5,np.uint8))
        cv2.imshow("nstate4",np.array((next_state[:,:,9:12]+1)*127.5,np.uint8))
    state = next_state'''
