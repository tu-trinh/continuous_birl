try:
    import Box2D
    from lunarlander_theta.envs.lunarlander_theta import LunarLanderTheta
except ImportError:
    Box2D = None
