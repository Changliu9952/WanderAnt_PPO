import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym


env = gym.make('AntPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('state size:', num_inputs,env.observation_space.low)
print('action size:', num_actions,env.action_space.high)

env.reset()
for t in range(1000):
    env.render()
    action = [0] * 8
    action = env.action_space.sample()
    # print('action', action)
    state, reward, done, info = env.step(action)
    # print('info:', info)

    # reward = forward velocity - sum(action^2) + live_bonus
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

# import torch.multiprocessing as mp
# import time
# processor =[]
# def func():
#     while True:
#         print('begin sleep')
#         time.sleep(1)
#         print('end sleep')
#
# for i in range(5):
#     p = mp.Process(target=func)
#     processor.append(p)
#
# for p in processor:
#     p.start()
#
# for p in processor:
#     p.join()