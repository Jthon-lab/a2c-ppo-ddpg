import tensorflow as tf
import numpy as np
import argparse
import gym
import threading

import network_utils as net_utils
import normalize_utils as norm_utils
import memory_utils as mem_utils
from ddpg_thread import Worker_Thread

from envs.toy_wrapper import TOY_Continuous
from envs.mujoco_wrapper import MUJOCO

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="HalfCheetah-v3")
    parser.add_argument("--env_seed",type=int,default=1000)
    parser.add_argument("--thread_num",type=int,default=8)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--q_lr_rate",type=float,default=5e-4)
    parser.add_argument("--pi_lr_rate",type=float,default=2e-4)
    parser.add_argument("--kstep",type=int,default=1)
    parser.add_argument("--update_freq",type=int,default=50)
    parser.add_argument("--minibatch_size",type=int,default=256)
    parser.add_argument("--start_size",type=int,default=10000)
    parser.add_argument("--buffer_size",type=int,default=200000)
    parser.add_argument("--noise_ratio",type=float,default=0.25)
    parser.add_argument("--tau",type=float,default=0.001)
    parser.add_argument("--total_timesteps",type=float,default=2e5)
    parser.add_argument("--timelimit",type=int,default=1000)
    parser.add_argument("--use_obs_norm",type=bool,default=True)
    args = parser.parse_known_args()[0]
    return args

args = get_args()
tf.set_random_seed(args.env_seed)
np.random.seed(args.env_seed)
def make_env_fn():
    #env = TOY_Continuous(args.env_name,args.env_seed,args.timelimit)
    env = MUJOCO(args.env_name,args.env_seed,args.timelimit)
    return env

env = make_env_fn()
ob_space = env.observation_space
ac_space = env.action_space
lock = threading.Condition()
session = tf.InteractiveSession()
memory = mem_utils.Memory_Buffer(args.thread_num,ob_space,ac_space,lock,args.buffer_size,args.kstep,args.minibatch_size)
obs_normalizer = norm_utils.Running_Estimator(ob_space.shape,lock,5)
rew_normalizer = norm_utils.Running_Reward_Normalizer((),lock,1)
if len(ob_space.shape) == 3: #use cnn
    hidden_sizes = (16,32,32,64,64,)
elif len(ob_space.shape) == 1:
    hidden_sizes = (64,64,64,)
else:
    raise NotImplementedError

threads = []
for i in range(args.thread_num):
    threads.append(Worker_Thread(i,args.thread_num,make_env_fn,session,lock,memory,args.use_obs_norm,obs_normalizer,
                                 hidden_sizes,args.minibatch_size,args.start_size,args.update_freq,args.kstep,args.gamma,
                                 args.pi_lr_rate,args.q_lr_rate,args.noise_ratio,args.tau,args.total_timesteps))

coord = tf.train.Coordinator()
worker_threads = []
for thread in threads:
    job = lambda:thread.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coord.join(worker_threads)
 
