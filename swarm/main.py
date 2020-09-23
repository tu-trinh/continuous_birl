from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p

env = SimpleEnv()
start_time = time.time()
curr_time = time.time() - start_time
while curr_time < 2400*np.pi:
    curr_time = time.time() - start_time
    x = 0.0
    y = 0.0
    events = p.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        if key == "w":
            x = 0.5
        if key == "a":
            y = 0.5
        if key == "d":
            y = 0
    action1 = [x, y]
    # action2 = [-0.5, 0]
    # action3 = [0.5, 0.5]
    next_state, reward, done, info = env.step(action1)#+action2+action3)
    # img = env.render()
    if done:
        break
env.close()
