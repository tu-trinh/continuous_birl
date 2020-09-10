try:
    import Box2D
    from LunarLander_c1.envs.lunar_lander_c1 import LunarLanderC1
except ImportError:
    Box2D = None