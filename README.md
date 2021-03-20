# a2c-ppo-ddpg
This repo is a simple implementation in tensorflow of some reinforcement learning algorithms, currently supporting **A2C**,**PPO** and **DDPG**.
The code based on [openai-baselines](https://github.com/openai/baselines) and [openai-spinning-up](https://github.com/openai/spinningup).
Details of A2C,PPO and DDPG can be found in papers:
- **A2C(Synchronous version of A3C) [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)** 
- **PPO(Proximal Policy Optimization) [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)** 
- **DDPG(Deep Determinstic Policy Gradient) [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971.pdf)** 

We test our implementation mainly in the following environment, which we provides a env wrapper for different types of environments:
- **ToyDiscrete(CartPole,MountainCar...)**
- **ToyContinuous(Pendulum,Acrobot...)**
- **MuJoCo(HalfCheetah,Walker2D,Ant,Swimmer...)**
- **Atari**
- **DMLab**

For MuJoCo and DMLab environment wrappers, you need to install the MuJoCo and DeepMind Lab in the computer according to the tutorials.
- MuJoCo-[https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
- DMLab-[https://github.com/deepmind/lab](https://github.com/deepmind/lab)

### Tricks ###
- Observation Normalization
- Advantage Normalization
- Generalized Advantage Estimation ([GAE](https://arxiv.org/pdf/1506.02438.pdf))
- Parallel Environments
- Policy Gradient Clipping

### TODOs ###
- Reward Clipping
- Value Function Normalization
- Parameter Clipping
- Adam Learning rate Annealing
- Improved version of DDPG(D4PG)
- ...
### Experimental Result ###


