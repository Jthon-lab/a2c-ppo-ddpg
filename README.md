# a2c-ppo-ddpg
This repo is a simple implementation in tensorflow of some reinforcement learning algorithms, currently supporting **A2C**,**PPO** and **DDPG**.
The codes based on [openai-baselines](https://github.com/openai/baselines) and [openai-spinning-up](https://github.com/openai/spinningup).
Details of A2C,PPO and DDPG can be found in papers:
**A2C(Synchronous version of A3C) [https://arxiv.org/pdf/1602.01783.pdf](https://arxiv.org/pdf/1602.01783.pdf)** 
**PPO(Proximal Policy Optimization) [https://arxiv.org/pdf/1707.06347.pdf](https://arxiv.org/pdf/1707.06347.pdf)** 
**DDPG(Deep Determinstic Policy Gradient) [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)** 

We test our implementation mainly in the following environment, which we provides a env wrapper for different types of environments:
**ToyDiscrete(CartPole,MountainCar...)**
**ToyContinuous(Pendulum,Acrobot...)**
**MuJoCo(HalfCheetah,Walker2D,Ant,Swimmer...)**
**Atari**
**DMLab**

For MuJoCo and DMLab environment wrappers, you need to install the MuJoCo and DeepMind Lab in the computer according to the tutorials.
MuJoCo-[https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
DMLab-[https://github.com/deepmind/lab](https://github.com/deepmind/lab)


