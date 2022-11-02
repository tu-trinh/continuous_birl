from env_1 import SimpleEnv
import time

N = 69

def random_swarm(num_cars = 1, theta = "obstacle"):
    return SimpleEnv(num_cars, theta)

def get_optimal_policy(env):
    n_cars = env.n_cars
    dataset = []

    xi = []
    cars = env.get_cars()
    actions = []
    for i in range(n_cars):
        action = [1.0, 0]
        actions.append(action)
    start_time = time.time()
    prev_time = 0.0
    first_pass = True
    run_time = time.time() - start_time
    while True:
        curr_time = time.time()
        time_step = curr_time - prev_time
        if time_step > 1.0:
            prev_time = curr_time
            for i in range(n_cars):
                if env.goal_reached(cars[i]):
                    actions[i] = [0.0, 0.0]
                else:
                    action = env.get_action(i)
                    actions[i] = action
        next_state, reward, done, info = env.step(actions)
        min_dists = env.get_min_dists()
        if time_step > 0.5:
            if first_pass:
                first_pass = False
            else:
                xi.append([curr_time]+[min_dists]+[next_state])
        state = next_state
        run_time = time.time() - start_time
        
        if done or run_time > 10.0:
            dataset.append(xi)
            break
    env.close()
    # print("Episode: {}\tTime taken: {}\tCollected Episodes: {}".format(episode, run_time, len(dataset)))
    return dataset

def generate_random_policies():
    pass

def generate_optimal_demo():
    pass

def calculate_expected_value_difference():
    pass

def comparison_grid():
    pass

def listify():
    pass