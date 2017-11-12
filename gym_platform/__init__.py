from gym.envs.registration import register

register(
    id='Platform-v0',
    entry_point='gym_platform.envs:PlatformEnv',
    max_episode_steps=500,
)

register(
    id='TwoLaneOneAgent-v0',
    entry_point='gym_platform.cooperativegames:TwoLaneOneAgentEnv',
    max_episode_steps=100,
)
register(
    id='TwoLane-v1',
    entry_point='gym_platform.cooperativegames:TwoLaneEnv',
    max_episode_steps=100,
)

