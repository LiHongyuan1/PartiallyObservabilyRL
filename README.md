# PartiallyObservabilyRL
This ia Partially Observabily Reinforcement Learning base on CFC network

# Introduction
We will train a CfC network to address a partially observable Markov decision process (POMDP). Specifically, we focus on a partially observable variant of the HalfCheetah environment in Mujoco. Additionally, we will evaluate the trained policy under noisy observation conditions to assess the robustness of the learned policy. This tutorial is designed as an introductory guide to using CfC networks in reinforcement learning, illustrating how to define a custom CfC network, integrate it with rllib, and showcase the benefits of RNNs in environments with partial observability.

# Environment
python == 2.10
pip install  -U    tensorflow==2.13.0   ncps==1.0.1     "ray[rllib]"==2.23.0     gymnasium[mujoco]     matplotlib imageio opencv-python

# example
<img width="723" height="723" alt="image" src="https://github.com/user-attachments/assets/c93437ec-49b3-441f-b854-05dd7e4a868d" />


