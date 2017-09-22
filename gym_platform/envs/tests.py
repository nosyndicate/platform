import platform_env
import gym
import time
import numpy as np

env = platform_env.PlatformEnv()
env.reset()
for index in range(1000):
    env.render()
    #print("rendered again " + str(_));
    stand_still = tuple([0, np.array([0]), np.array([0]), np.array([0])])
    state, r, done, other = env.step(env.action_space.sample())
    if done:
        env.reset()