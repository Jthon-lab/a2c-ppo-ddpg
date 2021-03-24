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

# DDPG is a value-iteration reinforcement learning algorithm
# By approximate Q(s,a) and backpropgating the gradient w.r.t value function and using chain rules to update policy network
# Similiar to DQN, it contains a target network (using soft update with tau' = (1-theta)*tau_old' + theta*tau) 
# For exploration, add Ornstein-Uhlenbeck noise into the action, so we need a class to generate OU noise
# for value loss using (Q_ph - Q)^2
# for update policy, using Q to maximize the q_function

class DDPG(object):
    def __init__(self,ob_space,ac_space,session,hidden_sizes=(64,64,64,),pi_lr_rate=1e-4,q_lr_rate=5e-4,tau=0.05,gamma=0.99,total_timesteps=1e6,max_grad_norm=0.5):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.session = session
        self.hidden_sizes = hidden_sizes
        self.act_dim = self.ac_space.shape[0]
        self.act_limit = self.ac_space.high[0]
        self.tau = tau
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

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
            self.q_ph = net_utils.placeholder(None)
    
    def _define_network_(self):
        with tf.variable_scope("eval_network",reuse=tf.AUTO_REUSE):
            if len(self.ob_space.shape) == 1:
                # use to take action
                with tf.variable_scope("pi",reuse=tf.AUTO_REUSE):
                    self.eval_policy = net_utils.mlp(self.x_ph,list(self.hidden_sizes)+list(self.ac_space.shape),activation=tf.nn.relu,output_activation=tf.nn.tanh)*self.act_limit
                # use to train the policy w.r.t maximize self.eval_policy_q
                with tf.variable_scope("q",reuse=tf.AUTO_REUSE):
                    self.eval_policy_q = net_utils.mlp(tf.concat((self.x_ph,self.eval_policy),axis=1),list(self.hidden_sizes)+[1],activation=tf.nn.relu,output_activation=None)
                # use to train the q_func w.r.t minimize (self.q_ph - self.eval_action_q)^2
                with tf.variable_scope("q",reuse=tf.AUTO_REUSE):
                    self.eval_action_q = net_utils.mlp(tf.concat((self.x_ph,self.a_ph),axis=1),list(self.hidden_sizes)+[1],activation=tf.nn.relu,output_activation=None)
            else:
                raise NotImplementedError
        with tf.variable_scope("target_network",reuse=tf.AUTO_REUSE):
            if len(self.ob_space.shape) == 1:
                # use to calc the target q_func Q(s',a=u(s'))
                with tf.variable_scope("pi",reuse=tf.AUTO_REUSE):
                    self.target_policy = net_utils.mlp(self.x_ph,list(self.hidden_sizes)+list(self.ac_space.shape),activation=tf.nn.relu,output_activation=tf.nn.tanh)*self.act_limit
                with tf.variable_scope("q",reuse=tf.AUTO_REUSE):
                    self.target_policy_q = net_utils.mlp(tf.concat((self.x_ph,self.target_policy),axis=1),list(self.hidden_sizes)+[1],activation=tf.nn.relu,output_activation=None)
            else:
                raise NotImplementedError
        
        self.pi_params = net_utils.get_vars("DDPG/eval_network/pi")
        self.q_params = net_utils.get_vars("DDPG/eval_network/q")
        self.eval_params = net_utils.get_vars("DDPG/eval_network")
        self.target_params = net_utils.get_vars("DDPG/target_network")

        self.target_update_op = [tf.assign(tp,self.tau*ep + (1-self.tau)*tp) for ep,tp in zip(self.eval_params,self.target_params)]
        self.target_init_op = [tf.assign(tp,ep) for ep,tp in zip(self.eval_params,self.target_params)]

    def _define_optimization_(self):
        with tf.variable_scope("optimization",reuse=tf.AUTO_REUSE):
            self.qvalue_optimizer = tf.train.AdamOptimizer(self.q_lr_rate,epsilon=1e-5)
            self.policy_optimizer = tf.train.AdamOptimizer(self.pi_lr_rate,epsilon=1e-5)
            self.policy_loss = -tf.reduce_mean(tf.squeeze(self.eval_policy_q,axis=1))
            self.qvalue_loss = tf.reduce_mean(tf.square(self.q_ph-tf.squeeze(self.eval_action_q,axis=1)))

            self.pi_grads = tf.gradients(self.policy_loss,self.pi_params)
            self.q_grads = tf.gradients(self.qvalue_loss,self.q_params)
            self.pi_grads,_ = tf.clip_by_global_norm(self.pi_grads,self.max_grad_norm)
            self.q_grads,_ = tf.clip_by_global_norm(self.q_grads,self.max_grad_norm)
            self.train_pi_op = self.policy_optimizer.apply_gradients(zip(self.pi_grads,self.pi_params))
            self.train_q_op = self.policy_optimizer.apply_gradients(zip(self.q_grads,self.q_params))
            #self.train_pi_op = self.policy_optimizer.minimize(self.policy_loss,var_list=self.pi_params) 
            #self.train_q_op = self.qvalue_optimizer.minimize(self.qvalue_loss,var_list=self.q_params)
    
    # using when calculating the target value function in training
    def get_target_q(self,x):
        q = self.session.run(self.target_policy_q,feed_dict={self.x_ph:x})
        return q
    # using when interacting with the environment
    def get_action(self,x):
        a = self.session.run(self.eval_policy,feed_dict={self.x_ph:[x]})
        return a[0]
    def init_target(self):
        self.session.run(self.target_init_op)
    def update(self,s1_batch,s2_batch,act_batch,rews_batch,dones_batch):
        q_values = self.get_target_q(s2_batch)*(1-dones_batch)
        rews_batch = np.concatenate((rews_batch,q_values),axis=1)
        q_targets = []
        for rews in rews_batch:
            q_targets.append(net_utils.discount_cumsum(rews,self.gamma)[0])
        q_targets = np.array(q_targets,np.float32)
        q_loss,_ = self.session.run([self.qvalue_loss,self.train_q_op],feed_dict={self.x_ph:s1_batch,self.a_ph:act_batch,self.q_ph:q_targets})
        self.session.run([self.train_pi_op,self.target_update_op],feed_dict={self.x_ph:s1_batch,self.a_ph:act_batch,self.q_ph:q_targets})
        #print("q_loss=%f"%q_loss)
    def save_model(self,ckpt_path):
        self.saver.save(self.session,ckpt_path)
    def load_model(self,ckpt_dir):
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        self.saver.restore(self.session,checkpoint)
    
