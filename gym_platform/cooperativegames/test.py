import gym
import gym_platform


env = gym.make('TwoLane-v1')
env.reset()

for index in range(1000):
    # Set the flag to render
    # Note: The env is wrapped in another env, so we need to go
    # one layer deeper.
    env.render()
    #print(env.action_space.sample())
    state, r, done, other = env.step(env.action_space.sample())
    assert(isinstance(other, dict))
    if done:
        env.reset()