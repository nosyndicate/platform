import platform_env
import gym
import time

env = platform_env.PlatformEnv()
env.reset()
for index in range(1000):
    env.render()
    #print("rendered again " + str(_));
    state, r, done, other = env.step(env.action_space.sample()) # take a random action
    # time.sleep(0.05)
    if done:
        env.reset()