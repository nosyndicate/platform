from gym.envs.registration import register

register(
    id='platform-v0',
    entry_point='gym_platform.envs:PlatformEnv',
)
