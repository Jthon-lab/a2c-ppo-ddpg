import tensorflow as tf
import os
import numpy as np
import scipy.signal
from gym.spaces import Box,Discrete

# reference to https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py

EPS = 1e-8
def make_dir(path):
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1,len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))
def placeholders(*args):
    return [placeholder(dim) for dim in args]
def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError
def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]
def get_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x,discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)
def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    entropy = tf.reduce_sum(-logp_all*tf.exp(logp_all),axis=1)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, entropy

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    entropy = log_std + 0.5*tf.log(2.0*np.pi*np.e)#(#tf.log()#tf.reduce_sum(log_std+.5*tf.log(2.0*pi*np.e),axis=1)
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi, entropy

def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):
    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
    with tf.variable_scope('pi'):
        pi, logp, logp_pi,entropy = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, entropy, v

def cnn(x,hidden_sizes=(32,),activation=tf.nn.tanh,output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.conv2d(x,h,kernel_size=5,strides=(2,2),padding="SAME",activation=activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.conv2d(x,hidden_sizes[-1],kernel_size=5,strides=(2,2),padding="SAME",activation=output_activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.layers.flatten(x)

def cnn_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
    with tf.variable_scope("feature"):
        feature = cnn(x,list(hidden_sizes),activation,activation)
    with tf.variable_scope("pi"):
        pi, logp, logp_pi,entropy = policy(feature, a, (), activation, output_activation, action_space)
    with tf.variable_scope("v"):
        v = tf.squeeze(mlp(feature,[1],activation,None),axis=1)
    return pi,logp,logp_pi,entropy,v


    






