import tensorflow as tf
import numpy as np
import gym
import os
import cv2
import network_utils as net_utils
import normalize_utils as norm_utils
import imageio
from memory_utils import Local_Memory
from ppo_model import PPO

from gym.spaces import Discrete,Box
from envs.toy_wrapper import TOY_Discrete
from envs.toy_wrapper import TOY_Continuous
from envs.mujoco_wrapper import MUJOCO
from envs.dmlab_wrapper import DMLab
from envs.atari_wrapper import ATARI

class Worker_Thread(object):
    def __init__(self,thread_id,thread_num,env_fn,session,lock,global_memory,obs_normalizer,
                 hidden_sizes=(64,64,64,),gamma=0.99,lam=0.95,
                 clip_ratio=0.2,nsteps=256,lr_rate=3e-4,value_coef=0.5,ent_coef=0.01,max_grad_norm=0.5,
                 train_iters=10,total_timesteps=2e6,ckpt_dir="./tmp/"):
        self.thread_id = thread_id
        self.thread_num = thread_num
        self.session = session
        self.lock = lock

        self.env = env_fn()
        self.env_name = self.env.env_name
        self.env_seed = self.env.env_seed
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space

        self.train_iters = train_iters
        self.total_timesteps = total_timesteps
        
        self.current_timesteps = 0
        self.current_episodes = 0

        #model path
        self.ckpt_dir = ckpt_dir + self.env_name + "-" + str(self.env_seed) + "/model/"
        net_utils.make_dir(self.ckpt_dir)
        self.ckpt_name = "ppo.ckpt"
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
        
        self.ppo = PPO(self.ob_space,self.ac_space,self.session,hidden_sizes,clip_ratio,value_coef,ent_coef,lr_rate,max_grad_norm,total_timesteps)
        self.local_memory = Local_Memory(self.ob_space,self.ac_space,nsteps,gamma,lam)
        self.obs_normalizer = obs_normalizer
        self.global_memory = global_memory

        if len(os.listdir(self.ckpt_dir))!=0:
            print("Train PPO from a checkpoint")
            self.session.run(tf.global_variables_initializer())
            self.ppo.load_model(self.ckpt_dir)
        else:
            print("Train PPO from scratch")
            self.session.run(tf.global_variables_initializer())
        
        self.train_epi_score = []
        self.train_step_score = []
    
    def synchorinize(self):
        if self.lock.acquire():
            worker_obs,worker_act,worker_adv,worker_ret,worker_logp = self.local_memory.get()
            self.global_memory.collect(worker_obs,worker_act,worker_adv,worker_ret,worker_logp)
            if self.global_memory.full():
                self.update()
                self.lock.notify_all()
            else:
                self.lock.wait()
            self.lock.release()
    
    def update(self):
        for i in range(self.train_iters):
            obs_batch,act_batch,adv_batch,ret_batch,logp_old_batch = self.global_memory.get()
            self.ppo.update(obs_batch,act_batch,adv_batch,ret_batch,logp_old_batch)
        self.global_memory.clear()
    
    def work(self):
        while self.current_timesteps < self.total_timesteps:
            self.current_episodes += 1
            done = False

            obs = self.env.reset()
            obs = self.obs_normalizer.normalize(obs)

            episode_score = 0
            episode_frames = []
            episode_rewards = []

            while not done:
                self.current_timesteps += 1
                if self.current_episodes%100 == 0 and self.thread_id == 0:
                    episode_frames.append(self.env.get_rgb())
                a,v_t,logp_t = self.session.run(self.ppo.get_action_ops,feed_dict={self.ppo.x_ph:np.expand_dims(obs,0)})
                
                next_obs,reward,done,info = self.env.step(a[0])
                next_obs = self.obs_normalizer.normalize(next_obs)

                self.local_memory.store(obs,a,reward,v_t,logp_t)
                if self.local_memory.full():
                    val = self.session.run(self.ppo.v,feed_dict={self.ppo.x_ph:np.expand_dims(next_obs,0)}) * (1-done)
                    self.local_memory.finish_path(val)
                    self.synchorinize()
                
                episode_score+=reward
                obs = next_obs
            
            self.train_epi_score.append([self.current_episodes,episode_score])
            self.train_step_score.append([self.current_timesteps,episode_score])
            self.local_memory.finish_path(0)

            if self.current_episodes%10 == 0:
                average_score = np.mean(np.array(self.train_epi_score)[:,1][-20:])
                max_score = np.max(np.array(np.array(self.train_epi_score)[:,1][-20:]))
                print("Worker:%d,episode %d:%d average score=%f,max_score=%f,frames=%d"%(self.thread_id,self.current_episodes-10,self.current_episodes,average_score,max_score,self.current_timesteps))
            if self.current_episodes%100 == 0:
                np.save(self.epi_score_path,np.array(self.train_epi_score))
                np.save(self.step_score_path,np.array(self.train_step_score))
                if self.thread_id == 0:
                    self.ppo.save_model(self.ckpt_path)
                    imageio.mimsave(self.video_dir + str(self.current_episodes) + ".gif",episode_frames,'GIF',duration=0.02)








        
