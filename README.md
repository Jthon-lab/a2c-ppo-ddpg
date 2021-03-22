# a2c-ppo-ddpg
This repo is a simple implementation in tensorflow of some reinforcement learning algorithms, currently supporting **A2C**,**PPO** and **DDPG**.
The code based on [openai-baselines](https://github.com/openai/baselines) and [openai-spinning-up](https://github.com/openai/spinningup).
Details of A2C,PPO and DDPG can be found in papers:
- **A2C(Synchronous version of A3C) [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)** 
- **PPO(Proximal Policy Optimization) [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)** 
- **DDPG(Deep Determinstic Policy Gradient) [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971.pdf)** 

We test our implementation mainly in the following environment, which we provides a env wrapper for different types of environments:
- ToyDiscrete(CartPole,MountainCar...)
- ToyContinuous(Pendulum,Acrobot...)
- MuJoCo(HalfCheetah,Walker2D,Ant,Swimmer...)
- Atari
- DMLab

For MuJoCo and DMLab environment wrappers, you need to install the MuJoCo and DeepMind Lab in the computer according to the tutorials.
- MuJoCo-[https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
- DMLab-[https://github.com/deepmind/lab](https://github.com/deepmind/lab)

### Tricks ###
- Observation Normalization
- Reward Scaling and Clipping 
- Advantage Normalization
- Generalized Advantage Estimation ([GAE](https://arxiv.org/pdf/1506.02438.pdf))
- Parallel Environments
- Policy Gradient Clipping

### TODOs ###
- Value Function Clipping
- Parameter Clipping
- Adam Learning rate Annealing
- Improved version of DDPG(D4PG)
- Orthogonal initialization and layer scaling
- Hyperbolic tan activations
- ...
### Experimental Result ###
1. ToyContinuous
2. MuJoCo
3. Atari
4. DMLab
For DeepMind Lab environment, we designed serveral basic and easy maze navigation tasks to test the performance of A2C and PPO, the top-down view of three different mazes and scores are shown below.

### Conclusion ###
- All of these RL algorithm are sensitive to hyper-parameters and need to fine-tune the parameters for different environments.
- Although in some environments, DDPG learns faster than A2C and PPO, but in most cases, PPO shows a more stable learning process without much delicate hyper-parameters settings.
- In sparse reward settings, pure RL algorithm may struggle to finish the task.
- Tricks are essential to the RL algorithms. 




