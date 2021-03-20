import numpy as np
import tensorflow as tf
import gym
import network_utils as net_utils
import normalize_utils as norm_utils

class DDPG(object):
    def __init__(self,ob_space,ac_space,session,hidden_sizes=(64,64,64,),pi_lr_rate=1e-4,q_lr_rate=1e-3,
                 tau=0.001,gamma=0.99,total_timesteps=1e6,max_grad_norm=0.5):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.session = session
        self.hidden_sizes = hidden_sizes

        self.act_dim = self.ac_space.shape[0]
        self.act_limit = self.ac_space.high[0]
        assert self.ac_space.high[0] == -self.ac_space.low[0] #symmetric action magnitude
        self.tau = tau
        self.gamma = gamma
        self.pi_lr_rate = pi_lr_rate
        self.q_lr_rate = q_lr_rate

        with tf.variable_scope("DDPG",reuse=tf.AUTO_REUSE):
            self._define_phs_()
            self._define_network_()
            self._define_optimization_()
        self.saver = tf.train.Saver(max_to_keep=1,var_list=self.pi_params+self.q_params)
    
    def _define_phs_(self):
        with tf.variable_scope("phs",reuse=tf.AUTO_REUSE):
            self.x_ph = net_utils.placeholder_from_space(self.ob_space)
            self.a_ph = net_utils.placeholder_from_space(self.ac_space)

            self.next_ph = net_utils.placeholder_from_space(self.ob_space)
            self.rews_ph = net_utils.placeholder(None)
            self.done_ph = net_utils.placeholder(None)
    
    def _define_network_(self):
        with tf.variable_scope("network",reuse=tf.AUTO_REUSE):
            with tf.variable_scope("eval",reuse=tf.AUTO_REUSE):
                with tf.variable_scope("pi",reuse=tf.AUTO_REUSE):
                    self.pi = net_utils.mlp(self.x_ph,hidden_sizes=list(self.hidden_sizes)+[self.ac_space.shape[0]],activation=tf.nn.leaky_relu,output_activation=tf.nn.tanh)*self.act_limit
                with tf.variable_scope("q",reuse=tf.AUTO_REUSE):
                    self.q_pi = net_utils.mlp(tf.concat((self.x_ph,self.pi),axis=1),hidden_sizes=list(self.hidden_sizes)+[1],activation=tf.nn.leaky_relu,output_activation=None)
                with tf.variable_scope("q",reuse=tf.AUTO_REUSE):
                    self.q_a = net_utils.mlp(tf.concat((self.x_ph,self.a_ph),axis=1),hidden_sizes=list(self.hidden_sizes)+[1],activation=tf.nn.leaky_relu,output_activation=None)
            with tf.variable_scope("target",reuse=tf.AUTO_REUSE):
                with tf.variable_scope("pi",reuse=tf.AUTO_REUSE):
                    self.target_pi = net_utils.mlp(self.next_ph,hidden_sizes=list(self.hidden_sizes)+[self.ac_space.shape[0]],activation=tf.nn.leaky_relu,output_activation=tf.nn.tanh)*self.act_limit
                with tf.variable_scope("q",reuse=tf.AUTO_REUSE):
                    self.target_q = net_utils.mlp(tf.concat((self.next_ph,self.target_pi),axis=1),hidden_sizes=list(self.hidden_sizes)+[1],activation=tf.nn.leaky_relu,output_activation=None)

        self.pi_params = net_utils.get_vars("DDPG/network/eval/pi")
        self.q_params = net_utils.get_vars("DDPG/network/eval/q") 
        self.eval_params = net_utils.get_vars("DDPG/network/eval")
        self.target_params = net_utils.get_vars("DDPG/network/target")
        self.init_target_op = [tf.assign(tp,ep) for tp,ep in zip(self.target_params,self.eval_params)]
        self.soft_update_op = [tf.assign(tp,self.tau*ep+(1-self.tau)*tp) for tp,ep in zip(self.target_params,self.eval_params)]
    
    def _define_optimization_(self):
        with tf.variable_scope("optimization",reuse=tf.AUTO_REUSE):
            self.pi_optimizer = tf.train.AdamOptimizer(self.pi_lr_rate)
            self.q_optimizer = tf.train.AdamOptimizer(self.q_lr_rate)
            self.pi_loss = -tf.reduce_mean(self.q_pi) # maximize the q function 

            self.target_return = tf.stop_gradient(self.rews_ph + self.gamma*(1-self.done_ph)*tf.squeeze(self.target_q,axis=1))
            self.q_loss = tf.reduce_mean(tf.square(self.target_return - tf.squeeze(self.q_a,axis=1)))

            self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss,var_list=self.pi_params)
            self.train_q_op = self.q_optimizer.minimize(self.q_loss,var_list=self.q_params)

    def get_action(self,x):
        a = self.session.run(self.pi,feed_dict={self.x_ph:[x]})
        return a[0]
    def init_target(self):
        self.session.run(self.init_target_op)
    def update(self,s1_batch,s2_batch,act_batch,rews_batch,dones_batch):
        q_loss,_ = self.session.run([self.q_loss,self.train_q_op],feed_dict={self.x_ph:s1_batch,
                                                                  self.a_ph:act_batch,
                                                                  self.next_ph:s2_batch,
                                                                  self.rews_ph:rews_batch,
                                                                  self.done_ph:dones_batch})
        pi_loss,_,_= self.session.run([self.pi_loss,self.train_pi_op,self.soft_update_op],feed_dict={self.x_ph:s1_batch,
                                                                                                     self.a_ph:act_batch,
                                                                                                     self.next_ph:s2_batch,
                                                                                                     self.rews_ph:rews_batch,
                                                                                                     self.done_ph:dones_batch})
        #print("q_loss=%f,pi_loss=%f"%(q_loss,pi_loss))
    def save_model(self,ckpt_path):
        self.saver.save(self.session,ckpt_path)
    def load_model(self,ckpt_dir):
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        self.saver.restore(self.session,checkpoint)

        
