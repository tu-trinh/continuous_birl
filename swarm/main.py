from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p
import cv2
import os
from PIL import Image

n_cars = 3
env = SimpleEnv(n_cars)
start_time = time.time()
curr_time = time.time() - start_time
x = 0.0
y = 0.0
cars = env.get_cars()
prev_time = 0.0
initial = True
prev_pos = []
while 1:
    actions = []
    curr_time = time.time()
    time_step = curr_time - prev_time
    if time_step > .5:
        prev_time = curr_time
        for i,car in enumerate(cars):
            if env.goal_reached(car):
                action = [0.0, 0.0]
            else:
                action = env.get_action(car)
                #escape out of local minima
                car_pos_orient = car.get_car_pos()
                car_pos = car_pos_orient[0]
                if initial:
                    prev_pos.append(car_pos)
                else:
                    dist = np.sqrt((prev_pos[i][0] - car_pos[0])**2\
                     + (prev_pos[i][1] - car_pos[1])**2)
                    if dist < 0.05:
                        action[0] = -1.0
                        action[1] = -action[1]
            actions.append(action)
            prev_pos[i] = car_pos
        # img = env.render()

    initial = False
    # actions = [[0., 0.]]
    next_state, reward, done, info = env.step(actions)
    # break
    if done:
        break

wait = input("Cars Reached Goal; Press Any Key To Exit")
env.close()
