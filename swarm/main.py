from env_1 import SimpleEnv
import numpy as np
import time

env = SimpleEnv()
start_time = time.time()
curr_time = time.time() - start_time
while curr_time < 2400*np.pi:
    curr_time = time.time() - start_time
    x = 0.0
    y = 0.0
    action1 = [x, y]
    # action2 = [-0.5, 0]
    # action3 = [0.5, 0.5]
    next_state, reward, done, info = env.step(action1)#+action2+action3)
    # img = env.render()
    if done:
        break
env.close()
