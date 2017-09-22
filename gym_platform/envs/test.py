import gym
import gym_platform


env = gym.make('Platform-v0')
env.reset()

for index in range(1000):
    # Set the flag to render
    # Note: The env is wrapped in another env, so we need to go
    # one layer deeper.
    env.env.call_render = True
    state, r, done, other = env.step(env.action_space.sample())
    if done:
        env.reset()