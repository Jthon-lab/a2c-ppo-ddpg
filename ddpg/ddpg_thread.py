import tensorflow as tf
import numpy as np
import gym
import os
import cv2
import network_utils as net_utils
import normalize_utils as norm_utils
import imageio
from ddpg_model import DDPG
from noise_utils import OrnsteinUhlenbeckActionNoise

from gym.spaces import Discrete,Box
from envs.toy_wrapper import TOY_Discrete
from envs.toy_wrapper import TOY_Continuous
from envs.mujoco_wrapper import MUJOCO
from envs.dmlab_wrapper import DMLab
from envs.atari_wrapper import ATARI

class Worker_Thread(object):
    def __init__(self,thread_id,thread_num,env_fn,session,lock,global_memory,
                 use_obs_norm,obs_normalizer,
                 hidden_sizes=(64,64,64,),batch_size=64, start_size=2000, update_freq = 50, stepk = 1,
                 gamma=0.99,pi_lr_rate=1e-4, q_lr_rate=5e-4, noise_ratio=0.1, tau=0.01,
                 total_timesteps=2e6,ckpt_dir="./tmp/"):
        self.thread_id = thread_id
        self.thread_num = thread_num
        self.session = session
        self.lock = lock

        self.env = env_fn()
        self.env_name = self.env.env_name
        self.env_seed = self.env.env_seed
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space

        self.action_low = self.ac_space.low
        self.action_high = self.ac_space.high
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_size = start_size
        self.update_freq = update_freq
        self.total_timesteps = total_timesteps
        
        self.stepk = stepk
        self.noise_ratio = noise_ratio
        self.noise_model = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.ac_space.shape,np.float32),sigma=0.3*np.ones(self.ac_space.shape,np.float32))
        self.noise_decay = noise_ratio / self.total_timesteps

        self.current_timesteps,self.current_episodes = 0,0

        #model path
        self.ckpt_dir = ckpt_dir + self.env_name + "-" + str(self.env_seed) + "/model/"
        net_utils.make_dir(self.ckpt_dir)
        self.ckpt_name = "ddpg.ckpt"
        self.ckpt_path = self.ckpt_dir + self.ckpt_name
        #score path
        self.epi_score_dir = ckpt_dir + self.env_name + "-" + str(self.env_seed) + "/epi_score/"
        net_utils.make_dir(self.epi_score_dir)
        self.epi_score_name = "worker_" + str(thread_id) +".npy"
        self.epi_score_path = self.epi_score_dir + self.epi_score_name
        #score path
        self.step_score_dir = ckpt_dir + self.env_name + "-" + str(self.env_seed) + "/step_score/"
        net_utils.make_dir(self.step_score_dir)
        self.step_score_name = "worker_" + str(thread_id) +".npy"
        self.step_score_path = self.step_score_dir + self.step_score_name
        #video path
        self.video_dir = ckpt_dir + self.env_name + "-" + str(self.env_seed) + "/video/"
        net_utils.make_dir(self.video_dir)
        self.ddpg = DDPG(self.ob_space,self.ac_space,self.session,hidden_sizes,pi_lr_rate,q_lr_rate,tau,gamma,total_timesteps)
        self.use_obs_norm = use_obs_norm
        self.obs_normalizer = obs_normalizer
        

        self.global_memory = global_memory

        if len(os.listdir(self.ckpt_dir))!=0:
            print("Train DDPG from a checkpoint")
            self.session.run(tf.global_variables_initializer())
            self.ddpg.load_model(self.ckpt_dir)
            self.ddpg.init_target()
        else:
            print("Train DDPG from scratch")
            self.session.run(tf.global_variables_initializer())
            self.ddpg.init_target()

        self.train_epi_score = []
        self.train_step_score = []

    def update(self):
        for i in range(self.update_freq):
            s1_batch,s2_batch,act_batch,rews_batch,dones_batch = self.global_memory.get_batch()
            self.ddpg.update(s1_batch,s2_batch,act_batch,rews_batch,dones_batch)

    def work(self):
        while self.current_timesteps < self.total_timesteps:
            self.current_episodes += 1
            done = False
            obs = self.env.reset()
            if self.use_obs_norm:
                obs = self.obs_normalizer.normalize(obs)
            #for k-step returns
            window_obs = []
            window_act = []
            window_reward = []

            episode_score = 0
            episode_frames = []
            episode_rewards = []

            while not done:
                self.current_timesteps += 1
                self.noise_ratio = np.clip(self.noise_ratio - self.noise_decay,0,1)
                if self.current_episodes%100 == 0 and self.thread_id == 0:
                    episode_frames.append(self.env.get_rgb())
                if self.current_timesteps<(self.start_size//self.thread_num):
                    a = np.clip((np.random.rand(self.ac_space.shape[0])-0.5)*2*self.action_high,self.action_low,self.action_high)
                else:
                    a = np.clip(self.ddpg.get_action(obs) + self.noise_ratio * self.noise_model(),self.action_low,self.action_high)
                next_obs,reward,done,info = self.env.step(a)
                if self.use_obs_norm:
                    next_obs = self.obs_normalizer.normalize(next_obs)
                #self.global_memory.store(obs,next_obs,a,[reward],done)
                #adjust window rewards to time step k
                    
                window_obs.append(obs)
                window_act.append(a)
                window_reward.append(reward)
                assert len(window_obs) == len(window_act) == len(window_reward)
                if len(window_reward) > self.stepk:
                    self.global_memory.store(window_obs[0],next_obs,window_act[0],np.array(window_reward,np.float32),done)
                    window_obs = window_obs[1:len(window_obs)]
                    window_act = window_act[1:len(window_act)]
                    window_reward = window_reward[1:len(window_reward)]
                
                if self.global_memory.current_size >= self.start_size and self.current_timesteps%self.update_freq == 0:
                    self.update()

                episode_rewards.append(reward)
                episode_score+=reward
                obs = next_obs
            
            self.train_epi_score.append([self.current_episodes,episode_score])
            self.train_step_score.append([self.current_timesteps,episode_score])
            if self.current_episodes%10 == 0:
                average_score = np.mean(np.array(self.train_epi_score)[:,1][-20:])
                max_score = np.max(np.array(np.array(self.train_epi_score)[:,1][-20:]))
                print("Worker:%d,episode %d:%d average score=%f,max_score=%f,frames=%d"%(self.thread_id,self.current_episodes-10,self.current_episodes,average_score,max_score,self.current_timesteps))
            
            if self.current_episodes%100 == 0:
                np.save(self.epi_score_path,np.array(self.train_epi_score))
                np.save(self.step_score_path,np.array(self.train_step_score))
                if self.thread_id == 0:
                    self.ddpg.save_model(self.ckpt_path)
                    imageio.mimsave(self.video_dir + str(self.current_episodes) + ".gif",episode_frames,'GIF',duration=0.02)

   