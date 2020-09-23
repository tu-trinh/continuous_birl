from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p

env = SimpleEnv()
start_time = time.time()
curr_time = time.time() - start_time
x = 0.0
y = 0.0
while curr_time < 2400*np.pi:
    curr_time = time.time() - start_time

    # p.connect(p.GUI)
    keys = env.get_key_events()
    for k, v in keys.items():
        if (k == 65298 and (v==1)):#6297
            x += -1.0
        if (k == 65296 and (v == 1)):
            y += 0.5
        if (k == 65295 and (v == 1)):
            y += -0.5
        if (k == 65297 and (v==1)):#6297
            x += 1.0
    if x > 1.0:
        x = 1.5
    elif x < -1.0:
        x = -1.5
    if y > 0.5:
        y = 0.5
    elif y < -0.5:
        y = -0.5
    print(keys.items())
    action1 = [x, y]
    # action2 = [-0.5, 0]
    # action3 = [0.5, 0.5]
    next_state, reward, done, info = env.step(action1)#+action2+action3)
    # img = env.render()
    if done:
        break
env.close()
