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
- Reward Scaling and Clipping (For PPO and A2C)
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
- **Hyper-parameter tuning**, for a better performance, we need time to select a proper hyper-parameters for each environments
- ...

### Experimental Result ###
1. **ToyEnvironments**
We run a2c,ppo and ddpg in CartPole, Acrobot and Pendulum environments as a starting task, and the learning process is shown below, each algorithm runs 3 random seed.
![ToyEnvironment](/figures/toy_curve.png)
2. **MuJoCo**
We run a2c,ppo and ddpg in HalfCheetah, Walker2d, Swimmer and Ant and each algorithm runs for 3 random seed. The learning curve is shown below.
![MuJoCoEnv](/figures/MuJoCo.png)
3. **Atari**
Not finished Yet.
4. **DMLab**
For DeepMind Lab environment, we designed serveral basic and easy maze navigation tasks to test the performance of A2C and PPO, the top-down view of three different mazes and scores are shown below. The agent was spawn in a fixed point and will be rewarded for +10 if it arrived at the destination point which is also fixed during training. In maze navigation task, the agent need to select action among {MOVE_FORWARD,TURN_LEFT,TURN_RIGHT} at each time step. In the complex maze navigation task, both algorithm failed to navigate to the target location due to limited exploration ability of them.
![MazeEnv](/figures/maze-topdown.png)
![MazeEnvResult](/figures/maze_curve.png)

### Conclusion ###
- All of these RL algorithm are sensitive to hyper-parameters and need to fine-tune the parameters for different environments.
- DDPG learns faster in some environment but suffers from high variance and unstable learning process, even fail to learn in some tasks
- In sparse reward settings, pure RL algorithm may struggle to finish the task.
- Tricks are essential to the RL algorithms. 




