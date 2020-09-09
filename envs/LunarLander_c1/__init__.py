from gym.envs.registration import registry, register, make, spec

register(
    id='LunarLanderC1-v0',
    entry_point='LunarLander_c1.envs:LunarLanderC1',
    max_episode_steps=1000,
    reward_threshold=200,
)
