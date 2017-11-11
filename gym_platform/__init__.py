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
    id='TwoLane-v0',
    entry_point='gym_platform.cooperativegames:TwoLaneEnv',
    max_episode_steps=100,
    kwargs={'sameSide' : True}, # means that the agents start at the same side of the environment
)

register(
    id='TwoLane-v1',
    entry_point='gym_platform.cooperativegames:TwoLaneEnv',
    max_episode_steps=100,
    kwargs={'sameSide' : False}, # means that they will be on oposite sides 
)

register(
    id='ReachPoint-v0',
    entry_point='gym_platform.cooperativegames:ReachPointEnv',
    max_episode_steps=20,
)