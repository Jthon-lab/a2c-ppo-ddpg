import tensorflow as tf
import numpy as np
import argparse
import gym
import threading

import network_utils as net_utils
import normalize_utils as norm_utils
import memory_utils as mem_utils
from ppo_thread import Worker_Thread

from envs.toy_wrapper import TOY_Discrete
from envs.toy_wrapper import TOY_Continuous
from envs.mujoco_wrapper import MUJOCO
from envs.dmlab_wrapper import DMLab
from envs.atari_wrapper import ATARI

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="u_maze_rl")
    parser.add_argument("--env_seed",type=int,default=1000)
    parser.add_argument("--thread_num",type=int,default=8)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--lam",type=float,default=0.97)
    parser.add_argument("--clip_ratio",type=float,default=0.2)
    parser.add_argument("--lr_rate",type=float,default=3e-4)
    parser.add_argument("--nsteps",type=int,default=256)
    parser.add_argument("--minibatch_size",type=int,default=128)
    parser.add_argument("--nepochs",type=int,default=10)
    parser.add_argument("--ent_coef",type=float,default=0.01)
    parser.add_argument("--value_coef",type=float,default=0.5)
    parser.add_argument("--max_grad_norm",type=float,default=0.5)
    parser.add_argument("--total_timesteps",type=float,default=1e6)
    parser.add_argument("--timelimit",type=int,default=299)
    parser.add_argument("--obs_norm",type=bool,default=True)
    parser.add_argument("--rew_norm",type=bool,default=True)
    args = parser.parse_known_args()[0]
    return args

args = get_args()
tf.set_random_seed(args.env_seed)
np.random.seed(args.env_seed)
def make_env_fn():
    #env = TOY_Discrete(args.env_name,args.env_seed,args.timelimit)
    #env = TOY_Continuous(args.env_name,args.env_seed,args.timelimit)
    env = MUJOCO(args.env_name,args.env_seed,args.timelimit)
    #env = DMLab(args.env_name,args.env_seed,args.timelimit)
    #env = ATARI(args.env_name,args.env_seed,args.timelimit,4,4,False,84)
    return env

env = make_env_fn()
ob_space = env.observation_space
ac_space = env.action_space
train_iters = int((args.nsteps * args.nepochs *args.thread_num) // args.minibatch_size)
lock = threading.Condition()
session = tf.InteractiveSession()
memory = mem_utils.Global_Memory(ob_space,ac_space,args.thread_num,args.minibatch_size)

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
    threads.append(Worker_Thread(i,args.thread_num,make_env_fn,session,lock,memory,
                                 args.obs_norm,obs_normalizer,args.rew_norm,rew_normalizer,
                                 hidden_sizes,args.gamma,args.lam,args.clip_ratio,args.nsteps,args.lr_rate,
                                 args.value_coef,args.ent_coef,args.max_grad_norm,train_iters,args.total_timesteps))


coord = tf.train.Coordinator()
worker_threads = []
for thread in threads:
    job = lambda:thread.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coord.join(worker_threads)



