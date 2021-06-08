from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p
import pickle

def get_optimals(episodes, theta):
    n_cars = 3
    dataset = []

    for episode in range(episodes * 5):
        xi = []
        env = SimpleEnv(n_cars, theta=theta)
        cars = env.get_cars()
        actions = []
        for i in range(n_cars):
            action = [1.0,0]
            actions.append(action)
        start_time = time.time()
        prev_time = 0.0
        first_pass = True
        run_time = time.time() - start_time
        while 1:
            curr_time = time.time()
            time_step = curr_time - prev_time
            if time_step > 1.0:
                prev_time = curr_time
                for i in range(n_cars):
                    action = env.get_action(i)        
                    actions[i] = action
                    if env.goal_reached(cars[i]):
                        actions[i] = [0.0, 0.0]
                # img = env.render()
            next_state, reward, done, info = env.step(actions)
            min_dists = env.get_min_dists()
            if time_step > 0.5:
                if first_pass:
                    first_pass = False
                else:
                    xi.append([curr_time]+[min_dists]+[next_state])
            state = next_state
            run_time = time.time() - start_time
            
            if done or run_time > 70.0:
                dataset.append(xi)
                break
        env.close()
        print("Episode: {}\tTime taken: {}\tCollected Episodes: {}"\
                    .format(episode, run_time, len(dataset)))
        if len(dataset) == episodes:
            break
    return dataset

def main():
    episodes = 10
    optimals = {'regular': [], 'goal': [], 'obstacle': []}
    optimals['regular'] = get_optimals(episodes, theta="regular")
    optimals['goal'] = get_optimals(episodes, theta="goal")
    optimals['obstacle'] = get_optimals(episodes, theta="obstacle")
    pickle.dump( optimals, open( "choices/optimal.pkl", "wb" ) )

if __name__ == "__main__":
    main()
  