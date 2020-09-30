from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p

n_cars = 3
env = SimpleEnv(n_cars)
start_time = time.time()
curr_time = time.time() - start_time
cars = env.get_cars()
initial = True
prev_pos = []
action_delay = 2.0
wait = input("Press Any Key To Start")
# Create actions list
actions = []
for i in range(n_cars):
    action = [0.5,0]
    actions.append(action)
i, current_car = env.get_closest_car()
curr_time = time.time()
delay_start_time = curr_time
prev_time = 0.0
for car in cars:
    car_pos_orient = car.get_car_pos()
    car_pos = car_pos_orient[0]
    prev_pos.append(car_pos)
while 1:
    curr_time = time.time()
    time_step = curr_time - prev_time
    if time_step > .5:
        prev_time = curr_time
        delay = curr_time - delay_start_time 
        if delay > action_delay:
            print(delay)
            i, current_car = env.get_closest_car()
            delay_start_time = time.time()
        action = env.get_action(current_car)
        car_pos_orient = current_car.get_car_pos()
        car_pos = car_pos_orient[0]
        dist = np.sqrt((prev_pos[i][0] - car_pos[0])**2\
         + (prev_pos[i][1] - car_pos[1])**2)
        if dist < 0.05:
            action[0] = action[0] * 2.0 / abs(action[0])
            action[1] = -action[1]
        actions[i] = action
        prev_pos[i] = car_pos

        # for i,car in enumerate(cars):
        #     if env.goal_reached(car):
        #         action = [0.0, 0.0]
        #     else:
        #         action = env.get_action(car)
        #         # action = [1.0, 0.0]
        #         #escape out of local minima
        #         car_pos_orient = car.get_car_pos()
        #         car_pos = car_pos_orient[0]
        #         if initial:
        #             prev_pos.append(car_pos)
        #         else:
        #             dist = np.sqrt((prev_pos[i][0] - car_pos[0])**2\
        #              + (prev_pos[i][1] - car_pos[1])**2)
        #             if dist < 0.05:
        #                 action[0] = action[0] * 2.0 / abs(action[0])
        #                 action[1] = -action[1]
        #     actions[i] = action
        #     prev_pos[i] = car_pos
        # img = env.render()
    next_state, reward, done, info = env.step(actions)
    # break
    if done:
        break

wait = input("Cars Reached Goal; Press Any Key To Exit")
env.close()
