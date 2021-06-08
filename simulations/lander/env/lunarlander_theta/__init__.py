from gym.envs.registration import registry, register, make, spec

register(
    id='LunarLanderTheta-v0',
    entry_point='lunarlander_theta.envs:LunarLanderTheta',
    max_episode_steps=1000,
    reward_threshold=200,
)
