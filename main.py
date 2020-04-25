import time, os
from ppo import PPO
import gym
import torch
import pybulletgym  # register PyBullet enviroments with open ai gym


def make_env():
	# env = gym.make('Humanoid-v1')
	env = gym.make('AntPyBulletEnv-v0')
	# env.render()
	return env

def main():
	# torch.set_default_tensor_type('torch.DoubleTensor')
	batchsz = 2048
	ppo = PPO(make_env, 10)

	# load model from checkpoint
	ppo.load()
	# comment this line to close evaluaton thread, to speed up training process.
	ppo.render(2)

	for i in range(10):
		print('iter:', i)

		ppo.update(batchsz)

		if i % 2 == 0 and i:
			ppo.save()


if __name__ == '__main__':
	print('make sure to execute: [export OMP_NUM_THREADS=1] already.')
	main()
