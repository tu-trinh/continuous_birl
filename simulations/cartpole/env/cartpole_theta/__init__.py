from gym.envs.registration import registry, register, make, spec

register(
    id='CartpoleTheta-v0',
    entry_point='cartpole_theta.envs:CartpoleTheta',
    max_episode_steps=200,
    reward_threshold=200,
)
