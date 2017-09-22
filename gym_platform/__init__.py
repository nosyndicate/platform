from gym.envs.registration import register

register(
    id='Platform-v0',
    entry_point='gym_platform.envs:PlatformEnv',
    max_episode_steps=500,
)
