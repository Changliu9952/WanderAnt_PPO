import time, os
import gym
import torch
import pybulletgym  # register PyBullet enviroments with open ai gym
import torch.multiprocessing as mp
import utils
import arguments
import models
from chief_worker import chief_worker
from dppo_agents import dppo_workers


# set OMP_NUM_THREADS=1, otherwise it will block multiprocessing threads.
os.environ['OMP_NUM_THREADS'] = '1'


def main():
	# get arguments
	args = arguments.args()

	# build environment
	env = gym.make(args.env_name)
	num_observations = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	# define the global network...
	critic_shared_model = models.Critic_Network(num_observations)
	critic_shared_model.share_memory()

	actor_shared_model = models.Actor_Network(num_observations, num_actions)
	actor_shared_model.share_memory()

	# define the traffic signal...
	traffic_signal = utils.TrafficLight()
	# define the counter
	critic_counter = utils.Counter()
	actor_counter = utils.Counter()
	# define the shared gradient buffer...
	critic_shared_grad_buffer = utils.Shared_grad_buffers(critic_shared_model)
	actor_shared_grad_buffer = utils.Shared_grad_buffers(actor_shared_model)
	# define shared observation state...
	shared_obs_state = utils.Running_mean_filter(num_observations)
	# define shared reward...
	shared_reward = utils.RewardCounter()
	# define the optimizer...
	critic_optimizer = torch.optim.Adam(critic_shared_model.parameters(), lr=args.value_lr)
	actor_optimizer = torch.optim.Adam(actor_shared_model.parameters(), lr=args.policy_lr)

	# prepare multiprocessing
	# find how many are available
	total_works = mp.cpu_count() - 1
	print(f'.....total available process  is {total_works}')
	num_of_workers = total_works
	print(f'.....we set num_of_processes to {num_of_workers}')

	processors = []
	workers = []

	# load model from check point
	pass
	#

	p = mp.Process(target=chief_worker, args=(num_of_workers, traffic_signal, critic_counter, actor_counter,
											  critic_shared_model, actor_shared_model, critic_shared_grad_buffer,
											  actor_shared_grad_buffer,
											  critic_optimizer, actor_optimizer, shared_reward, shared_obs_state,
											  args.policy_update_step, args.env_name))

	processors.append(p)

	for idx in range(num_of_workers):
		workers.append(dppo_workers(args))

	for worker in workers:
		p = mp.Process(target=worker.train_network, args=(traffic_signal, critic_counter, actor_counter,
														  critic_shared_model, actor_shared_model, shared_obs_state,
														  critic_shared_grad_buffer, actor_shared_grad_buffer,
														  shared_reward))
		processors.append(p)

	for p in processors:
		p.start()

	for p in processors:
		p.join()








if __name__ == '__main__':
	print('make sure to execute: [export OMP_NUM_THREADS=1] already.')
	main()
