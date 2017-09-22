import platform_env
import gym
import time
import numpy as np

env = platform_env.PlatformEnv()
env.reset()
for index in range(1000):
    env.render()
    #print("rendered again " + str(_));
    # stand still
    # action = tuple([0, np.array([0]), np.array([0]), np.array([0])])
    # hop
    # action = tuple([1, np.array([0]), np.array([3*index]), np.array([-4*index])])
    # state, r, done, other = env.step(action)
    state, r, done, other = env.step(env.action_space.sample())
    # time.sleep(0.05)
    if done:
        env.reset()