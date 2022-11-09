import gym
import numpy as np
import pickle

random_policies = []
beta = 10
possible_rewards = ["center", "anywhere", "crash"]

def reward(policy, theta):
    R = 0
    shaping = 0
    prev_shaping = None
    reward = 0
    initial_waypoint = policy[0]
    initial_state = initial_waypoint[3]
    initial_x = initial_state[0]
    for i, waypoint in enumerate(policy):
        state = waypoint[3]
        action = waypoint[1]
        if theta == "center":
            shaping = \
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        elif theta == "anywhere":
            shaping = \
                - 100*np.sqrt(state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        elif theta == "crash":
            shaping = \
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 0.0*abs(state[4]) + 0.0*state[6] + 0.0*state[7]

        if prev_shaping is not None:
            reward = shaping - prev_shaping
        prev_shaping = shaping

        if action == 1 or 3:
            reward -= 0.03

        elif action == 2:
            reward -= 0.30

        awake = waypoint[2]
        if i == len(waypoint) and awake:
            if np.linalg.norm(state[0]) < 0.1 and theta == "center":
                reward = +100
            if np.linalg.norm(state[0] - initial_x) < 0.1 and theta == "anywhere":
                reward = +100
            else:
                reward = -100
        elif not awake:
            if theta == "crash":
                reward = +100
            else:
                reward = -100
        R += reward
    return R / 100
    

def random_lander():
    env = gym.make('LunarLanderTheta-v0')
    return env

def get_optimal_policy(theta):
    # Theta options: center, anywhere, crash
    policies = pickle.load(open("choices/optimal.pkl", "rb"))
    return policies[theta][0]

def generate_random_policies():
    noisies = pickle.load(open("choices/noisies_set.pkl", "rb"))
    for policy in noisies[0][:5]:
        random_policies.append(policy)

def generate_optimal_demos(num_demos):
    # demos = pickle.load(open("choices/demos.pkl", "rb"))
    demos = pickle.load(open("choices/optimal.pkl", "rb"))["center"]
    return demos[:num_demos]

def calculate_expected_value_difference(eval_policy, opt_policy, theta, rn = False):
    V_eval = reward(eval_policy, theta)
    V_opt = reward(opt_policy, theta)
    if rn:
        V_rand = 0
        for random_policy in random_policies:
            V_rand += reward(random_policy, theta)
        V_rand /= len(random_policies)
        evd = (V_opt - V_eval) / (V_opt - V_rand)
    else:
        evd = V_opt - V_eval
    return evd

def comparison_grid(possible_policies):
    # possible_policies = [get_optimal_policy(pr, env.lava) for pr in possible_rewards]
    values = [[0 for j in range(len(possible_rewards))] for i in range(len(possible_policies))]
    for i in range(len(possible_policies)):
        for j in range(len(possible_rewards)):
            value = reward(possible_policies[i], possible_rewards[j])
            # print("For theta_{} = {} and policy {}, cost is {}".format(j, possible_rewards[j], i, value))
            values[i][j] = value
    return np.array(values)

def listify(arr, policy = True):
    if policy:
        pi = []
        for timestep in arr:
            pi.append(timestep[1])
        return pi
    else:
        grid = []
        for row in arr:
            grid.append(list(row))
        return grid

def birl(demos):
    probs = []
    counters = []
    for theta in possible_rewards:
        counters.append(get_optimal_policy(theta))
    for theta in possible_rewards:
        demo_reward = np.array([reward(demo, theta) for demo in demos], dtype = np.float32)
        counter_reward = np.array([reward(demo, theta) for demo in counters], dtype = np.float32)
        n = np.exp(beta * sum(demo_reward))
        d = sum(np.exp(beta * counter_reward)) ** len(demos)
        probs.append(n/d)
    Z = sum(probs)
    pmf = np.asarray(probs) / Z
    return pmf, possible_rewards[np.argmax(pmf)], counters[np.argmax(pmf)]