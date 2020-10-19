from env_1 import SimpleEnv
import numpy as np
import time
import pybullet as p
import pickle

def get_human(episodes, t_delay, type):
    n_cars = 3
    dataset = []
    if type == "regular":
        noise_threshold = 0.0
    elif type == "noise":
        noise_threshold = 0.2
    elif type == "counterfactual":
        noise_threshold = 0.0

    for episode in range(episodes * 5):
        if type == "counterfactual":
            stoptime = np.random.randint(6, 30)
        else:
            stoptime = 120
        xi = []
        env = SimpleEnv(n_cars)
        cars = env.get_cars()
        actions = []
        for i in range(n_cars):
            action = [1.0,0]
            actions.append(action)
        closest_car_index, current_car = env.get_closest_car()
        start_time = time.time()
        delay_start_time = start_time
        prev_time = 0.0
        first_pass = True
        run_time = time.time() - start_time
        while 1:
            curr_time = time.time()
            time_step = curr_time - prev_time
            if time_step > 1.0:
                prev_time = curr_time
                delay = curr_time - delay_start_time
                if delay > t_delay:
                    closest_car_index, current_car = env.get_closest_car()
                    delay_start_time = time.time()

                for i in range(n_cars):
                    if i == closest_car_index:
                        action = env.get_action(i)        
                        if np.random.random() < noise_threshold:
                            action[0] = np.random.choice([-1.5,2.0])
                            action[1] = np.random.choice([-0.5,0.5])
                        actions[i] = action
                    if env.goal_reached(cars[i]) or (run_time > stoptime):
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
            
            if run_time > 70.0:
                if not (type == "regular"):
                    dataset.append(xi)
                break
            # break
            if done:
                dataset.append(xi)
                break
        env.close()
        print("Episode: {}\tTime taken: {}\tCollected Episodes: {}"\
                    .format(episode, run_time, len(dataset)))
        if len(dataset) == episodes:
            break
    return dataset

def main():
    episodes = 5
    t_delay = 4
    demos = get_human(episodes, t_delay, type="regular")
    pickle.dump( demos, open( "choices/demos.pkl", "wb" ) )
    print(demos)
    print("Demos complete")
    noisies = get_human(10, t_delay=t_delay, type="noise")
    pickle.dump( noisies, open( "choices/noisy.pkl", "wb" ) )
    print("noisies complete")
    counterfactuals = get_human(10, t_delay=t_delay, type="counterfactual")
    pickle.dump( counterfactuals, open( "choices/counterfactual.pkl", "wb" ) )
    print("counterfactual complete")

if __name__ == "__main__":
    main()
  