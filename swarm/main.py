from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p

n_cars = 3
env = SimpleEnv(n_cars)
cars = env.get_cars()
action_delay = 4
wait = input("Press Any Key To Start")
# Create actions list
actions = []
for i in range(n_cars):
    action = [1.0,0]
    actions.append(action)
closest_car_index, current_car = env.get_closest_car()
curr_time = time.time()
delay_start_time = curr_time
prev_time = 0.0

while 1:
    curr_time = time.time()
    time_step = curr_time - prev_time
    if time_step > .5:
        prev_time = curr_time
        delay = curr_time - delay_start_time
        if delay > action_delay:
            closest_car_index, current_car = env.get_closest_car()
            delay_start_time = time.time()

        for i in range(n_cars):
            if i == closest_car_index:
                action = env.get_action(i)        
                actions[i] = action
            if env.goal_reached(cars[i]):
                actions[i] = [0.0, 0.0]

        # img = env.render()
    # print(actions)
    next_state, reward, done, info = env.step(actions)
    # break
    if done:
        break

wait = input("Cars Reached Goal; Press Any Key To Exit")
env.close()
