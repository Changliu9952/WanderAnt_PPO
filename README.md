# Wander Ant
Author by: Chang, Yihang, Siyuan
##Todos
* Build skelenton for code
* finish basic guidance for reproduce project

* inital env
* set torch seeds
* set multiprocess
for
sample trajectories
loss function 
backprop


## Introduction
An PPO implementation with Pybullet

## Dependence
Please walk through following step to install all dependencies of the project:

Configure a virtual environment with python 3.7 (optional)
 ###Intall Installing Pybullet-Gym 
* Perform a minimal installation of OpenAI Gym with
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
* install Pybullet-Gym by cloning the repository and install locally
```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```
Important Note: Do not use python setup.py install as this will not copy the assets (you might get missing SDF file errors).
* Finally, to test installation, open python and run:
```
import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym

env = gym.make('HumanoidPyBulletEnv-v0')
# env.render() # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked
```

### Install other packages



