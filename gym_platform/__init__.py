from gym.envs.registration import register

register(
    id='Platform-v0',
    entry_point='gym_platform.envs:PlatformEnv',
    max_episode_steps=500,
)

register(
    id='Narrow-v0',
    entry_point='gym_platform.cooperativegames:NarrowEnv',
    max_episode_steps=100,
)


register(
    id='ReachPoint-v0',
    entry_point='gym_platform.cooperativegames:ReachPointEnv',
    max_episode_steps=20,
)