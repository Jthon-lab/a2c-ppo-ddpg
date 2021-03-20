import numpy as np
import tensorflow as tf
import gym
import network_utils as net_utils
import normalize_utils as norm_utils
from gym.spaces import Discrete,Box
#TODO: Add learning rate decay
#TODO: Add support for LSTM/GRU
#TODO: Add layer_normalization or batch_normalization
#TODO: Add PPO clip parameter annealing
#Some Details : when using mlp_ac architecture, the layers don't share
#parameters while using cnn_ac architecture, policy and value share the top convnet 
class A2C(object):
    def __init__(self,ob_space,ac_space,session,hidden_sizes=(64,64,64,),ent_coef=0.01,
                 lr_rate=3e-4,max_grad_norm=0.5,total_timesteps=1e6):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hidden_sizes = hidden_sizes
        self.session = session

        self.e_coef = ent_coef
        self.lr_rate = lr_rate
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps

        with tf.variable_scope("A2C",reuse=tf.AUTO_REUSE):
            self._define_phs_()
            self._define_network_()
            self._define_optimization_()
        self.saver = tf.train.Saver(max_to_keep=1,var_list=self.params)

    def _define_phs_(self):
        with tf.variable_scope("phs",reuse=tf.AUTO_REUSE):
            self.x_ph = net_utils.placeholder_from_space(self.ob_space)
            self.a_ph = net_utils.placeholder_from_space(self.ac_space)
            self.ret_ph = net_utils.placeholder(None)
            self.adv_ph = net_utils.placeholder(None)
    def _define_network_(self):
        with tf.variable_scope("network",reuse=tf.AUTO_REUSE):
            if len(self.ob_space.shape) == 3: #use convnet
                self.pi,self.logp,self.logp_pi,self.pi_entropy,self.v = net_utils.cnn_actor_critic(self.x_ph,self.a_ph,self.hidden_sizes,activation=tf.nn.tanh,action_space=self.ac_space)
            elif len(self.ob_space.shape) == 1: #use mlpnet
                self.pi,self.logp,self.logp_pi,self.pi_entropy,self.v = net_utils.mlp_actor_critic(self.x_ph,self.a_ph,self.hidden_sizes,activation=tf.nn.tanh,action_space=self.ac_space) 
        self.get_action_ops = [self.pi,self.v]                
        self.params = net_utils.get_vars("A2C/network")

    def _define_optimization_(self):
        #ppo restrict the difference between the old policy and the current policy
        with tf.variable_scope("optimization",reuse=tf.AUTO_REUSE):
            self.pi_loss = -tf.reduce_mean(self.adv_ph*self.logp)
            self.vi_loss = tf.reduce_mean(tf.square(self.ret_ph - self.v))
            self.ent_loss = tf.reduce_mean(self.pi_entropy)

            self.pi_loss = self.pi_loss - self.e_coef*self.ent_loss
            self.optimizer = tf.train.AdamOptimizer(self.lr_rate,epsilon=1e-5)
            self.pi_grads = tf.gradients(self.pi_loss,self.params)
            self.vi_grads = tf.gradients(self.vi_loss,self.params)
            self.pi_grads,_ = tf.clip_by_global_norm(self.pi_grads,self.max_grad_norm)
            self.vi_grads,_ = tf.clip_by_global_norm(self.vi_grads,self.max_grad_norm)
            self.train_pi_op = self.optimizer.apply_gradients(zip(self.pi_grads,self.params))
            self.train_vi_op = self.optimizer.apply_gradients(zip(self.vi_grads,self.params))

    def save_model(self,ckpt_path):
        self.saver.save(self.session,ckpt_path)
    def load_model(self,ckpt_dir):
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        self.saver.restore(self.session,checkpoint)
    def update_pi(self,obs_batch,act_batch,adv_batch,ret_batch):
        feed_dict = {self.x_ph:obs_batch,
                     self.a_ph:act_batch,
                     self.adv_ph:adv_batch,
                     self.ret_ph:ret_batch}
        _,pi_loss,ent_loss = self.session.run([self.train_pi_op,self.pi_loss,self.ent_loss],feed_dict=feed_dict)
    def update_vi(self,obs_batch,act_batch,adv_batch,ret_batch):
        feed_dict = {self.x_ph:obs_batch,
                     self.a_ph:act_batch,
                     self.adv_ph:adv_batch,
                     self.ret_ph:ret_batch}
        _,vi_loss = self.session.run([self.train_vi_op,self.vi_loss],feed_dict=feed_dict)
