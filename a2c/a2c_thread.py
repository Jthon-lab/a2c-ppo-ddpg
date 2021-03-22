import tensorflow as tf
import numpy as np
import gym
import os
import cv2
import network_utils as net_utils
import normalize_utils as norm_utils
import imageio
from memory_utils import Local_Memory
from a2c_model import A2C

from gym.spaces import Discrete,Box
from envs.toy_wrapper import TOY_Discrete
from envs.toy_wrapper import TOY_Continuous
from envs.mujoco_wrapper import MUJOCO
from envs.dmlab_wrapper import DMLab
from envs.atari_wrapper import ATARI

#Different from actor-critic methods, vanilla policy gradient don't use TD-error to train the value-function,
#But use value-function to estimate the expection of the Reward-To-Go in an episode giving a state
#Basically, VPG can only update pi(s) for one iteration with out the importance sampling,but we can test the performance with
#multiple update times to see what will happen
class Worker_Thread(object):
    def __init__(self,thread_id,thread_num,env_fn,session,lock,global_memory,
                 use_obs_norm,obs_normalizer,use_rew_norm,rew_normalizer,
                 hidden_sizes=(64,64,64,),gamma=0.99,lam=0.95,nsteps=256,lr_rate=3e-4,
                 ent_coef=0.01,max_grad_norm=0.5,train_vi_iters=20,
                 train_pi_iters=1,total_timesteps=2e6,ckpt_dir="./tmp/"):
        
        self.thread_id = thread_id
        self.thread_num = thread_num
        self.session = session
        self.lock = lock

        self.env = env_fn()
        self.env_name = self.env.env_name
        self.env_seed = self.env.env_seed
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space

        self.train_vi_iters = train_vi_iters
        self.train_pi_iters = train_pi_iters
        self.total_timesteps = total_timesteps
        
        self.current_timesteps = 0
        self.current_episodes = 0
        self.gamma = gamma

        #model path
        self.ckpt_dir = ckpt_dir + self.env_name + "-" + str(self.env_seed) + "/model/"
        net_utils.make_dir(self.ckpt_dir)
        self.ckpt_name = "vpg.ckpt"
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
        
        self.a2c = A2C(self.ob_space,self.ac_space,self.session,hidden_sizes,ent_coef,lr_rate,max_grad_norm,total_timesteps)
        self.local_memory = Local_Memory(self.ob_space,self.ac_space,nsteps,gamma,lam)
        self.use_obs_norm = use_obs_norm
        self.use_rew_norm = use_rew_norm
        self.obs_normalizer = obs_normalizer
        self.rew_normalizer = rew_normalizer
        self.global_memory = global_memory

        if len(os.listdir(self.ckpt_dir))!=0:
            print("Train A2C from a checkpoint")
            self.session.run(tf.global_variables_initializer())
            self.a2c.load_model(self.ckpt_dir)
        else:
            print("Train A2C from scratch")
            self.session.run(tf.global_variables_initializer())
        
        self.train_epi_score = []
        self.train_step_score = []
    
    def synchorinize(self):
        if self.lock.acquire():
            worker_obs,worker_act,worker_adv,worker_ret = self.local_memory.get()
            self.global_memory.collect(worker_obs,worker_act,worker_adv,worker_ret)
            if self.global_memory.full():
                self.update()
                self.lock.notify_all()
            else:
                self.lock.wait()
            self.lock.release()
    
    def update(self):
        for i in range(self.train_pi_iters):
            obs_batch,act_batch,adv_batch,ret_batch = self.global_memory.get()
            self.a2c.update_pi(obs_batch,act_batch,adv_batch,ret_batch)
        for i in range(self.train_vi_iters):
            obs_batch,act_batch,adv_batch,ret_batch = self.global_memory.get()
            self.a2c.update_vi(obs_batch,act_batch,adv_batch,ret_batch)
        self.global_memory.clear()
    
    def work(self):
        while self.current_timesteps < self.total_timesteps:
            self.current_episodes += 1
            done = False
            obs = self.env.reset()
            if self.use_obs_norm:
                obs = self.obs_normalizer.normalize(obs)

            episode_score = 0
            episode_frames = []
            episode_rewards = []

            while not done:
                self.current_timesteps += 1
                if self.current_episodes%100 == 0 and self.thread_id == 0:
                    episode_frames.append(self.env.get_rgb())
                
                a,v_t = self.session.run(self.a2c.get_action_ops,feed_dict={self.a2c.x_ph:np.expand_dims(obs,0)})
                next_obs,reward,done,info = self.env.step(a[0])
                if self.use_obs_norm:
                    next_obs = self.obs_normalizer.normalize(next_obs)

                if self.use_rew_norm:
                    self.local_memory.store(obs,a,self.rew_normalizer.normalize_without_mean(reward),v_t)
                else:
                    self.local_memory.store(obs,a,reward,v_t)

                if self.local_memory.full():
                    val = self.session.run(self.a2c.v,feed_dict={self.a2c.x_ph:np.expand_dims(next_obs,0)}) * (1-done)
                    self.local_memory.finish_path(val)
                    self.synchorinize()
                
                episode_rewards.append(reward)
                episode_score+=reward
                obs = next_obs
            
            self.rew_normalizer.store(episode_rewards,self.gamma)
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
                    self.a2c.save_model(self.ckpt_path)
                    imageio.mimsave(self.video_dir + str(self.current_episodes) + ".gif",episode_frames,'GIF',duration=0.02)








        
