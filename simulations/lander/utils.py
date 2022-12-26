# import gym
import numpy as np
import pickle
import random

random_policies = []
beta = 10
num_features = 11

all_hypotheses = {}
main_hypotheses = {
    'center': [0, 0, -100, 0, 0, -100, 0, -100, 0, 10, 10],
    'anywhere': [0, 0, -200, 0, 0, -50, 0, -200, 0, 5, 5],
    'crash': [0, 0, -300, 0, 0, -10, 0, -300, 0, 8, 8]
}
alt_hypotheses = {}
f = open("hypotheses.txt", "r")
for line in f:
    alt_hypotheses.update(eval(line))
all_hypotheses.update(main_hypotheses)
all_hypotheses.update(alt_hypotheses)

hypo_policies = pickle.load(open("hypothesis_policies.pkl", "rb"))
main_policies = pickle.load(open("choices/optimal.pkl", "rb"))

failed_hypotheses = ['hypo3', 'hypo8', 'hypo9', 'hypo11', 'hypo12', 'hypo13', 'hypo14', 'hypo15', 'hypo16', 'hypo17', 'hypo18', 'hypo20', 'hypo21', 'hypo24', 'hypo28', 'hypo29', 'hypo32', 'hypo33', 'hypo36', 'hypo38', 'hypo41', 'hypo43', 'hypo44', 'hypo46', 'hypo49', 'hypo51', 'hypo54', 'hypo59', 'hypo60', 'hypo62', 'hypo64', 'hypo65', 'hypo68', 'hypo70', 'hypo72', 'hypo77', 'hypo78', 'hypo80', 'hypo81', 'hypo83', 'hypo84', 'hypo85', 'hypo87', 'hypo88', 'hypo90', 'hypo91', 'hypo100', 'hypo102', 'hypo110', 'hypo111', 'hypo113', 'hypo119', 'hypo120', 'hypo122', 'hypo126', 'hypo129', 'hypo133']

# Everything is wrapped in one big array.
# Each episode, aka policy/trajectory, is an array inside here.
# Each episode consist of arrays of the form [timestep, action, awake, state] (state itself is also an array).

"""
Lander class
"""
class Lander:
    def __init__(self, tt, fw):
        self.true_theta = tt
        self.feature_weights = fw

    def set_rewards(self, _feature_weights):
        self.feature_weights = _feature_weights


"""
Utility functions
"""
def reward(policy, theta):
    R = 0
    shaping = 0
    prev_shaping = None
    reward = 0
    initial_waypoint = policy[0]
    initial_state = initial_waypoint[3]
    initial_x = initial_state[0]
    for i, waypoint in enumerate(policy):
        action = waypoint[1]
        awake = waypoint[2]
        state = waypoint[3]
        features = np.array([
            state[0],
            state[1],
            np.sqrt(state[0]**2 + state[1]**2),
            state[2],
            state[3],
            np.sqrt(state[2]**2 + state[3]**2),
            state[4],
            np.abs(state[4]),
            state[5],
            state[6],
            state[7]
        ])
        theta_vec = all_hypotheses[theta]
        theta_vec = theta_vec / np.linalg.norm(theta_vec)
        
        shaping = np.dot(theta_vec, features)
        if np.isnan(shaping):
            shaping = 0.0001
        if prev_shaping is not None:
            reward = shaping - prev_shaping
        prev_shaping = shaping
        # if action == 1 or 3:
        #     reward -= 0.03
        # elif action == 2:
        #     reward -= 0.30
        R += reward
    return R / 100
    
def random_lander(true_theta):
    # features
    # s[0], s[1], sqrt(s[0]^2 + s[1]^2), s[2], s[3], sqrt(s[2]^2 + s[3]^2),
    # s[4], abs(s[4]), s[5], s[6], s[7], awake, not awake

    # reference
    # center: np.array([0, 0, -100, 0, 0, -100, 0, -100, 0, 10, 10, ±100, ±100])
    # anywhere: np.array([0, -100, 0, 0, 0, -100, 0, -100, 0, 10, 10, ±100, ±100])
    # crash: np.array([0, 0, -100, 0, 0, -100, 0, 0, 0, 0, 0, ±100, ±100])

    if true_theta == "center":
        feature_weights = np.array([0, 0, random.randint(-120, -80), 0, 0, random.randint(-120, -80), 0, random.randint(-120, -80), 0, random.randint(1, 20), random.randint(1, 20)])
    elif true_theta == "anywhere":
        feature_weights = np.array([0, random.randint(-120, -80), 0, 0, 0, random.randint(-120, -80), 0, random.randint(-120, -80), 0, random.randint(1, 20), random.randint(1, 20)])
    elif true_theta == "crash":
        feature_weights = np.array([0, 0, random.randint(-120, -80), 0, 0, random.randint(-120, -80), 0, 0, 0, 0, 0])
    env = Lander(true_theta, feature_weights)
    return env

def get_optimal_policy(theta, index = 0):
    # Theta options: center, anywhere, crash
    if theta == "center" or theta == "anywhere" or theta == "crash":
        return main_policies[theta][index]
    else:
        try:
            policy = hypo_policies[theta][index]
        except IndexError:
            policy = hypo_policies[theta][-1]
        return policy

def generate_random_policies():
    noisies = pickle.load(open("choices/noisies_set.pkl", "rb"))
    for policy in noisies[0][:5]:
        random_policies.append(policy)

def generate_optimal_demos(num_demos, theta = "center"):
    # demos = pickle.load(open("choices/demos.pkl", "rb"))
    if theta == "center" or theta == "anywhere" or theta == "crash":
        demos = main_policies[theta]
    else:
        demos = hypo_policies[theta]
    return demos[:num_demos]

def expected_feature_counts(trajectory):
    feature_counts = np.array([0 for _ in range(num_features)])
    for waypoint in trajectory:
        state = waypoint[3]
        features = np.array([state[0], state[1], np.sqrt(state[0]**2 + state[1]**2), state[2], state[3], np.sqrt(state[2]**2 + state[3]**2), state[4], np.abs(state[4]), state[5], state[6], state[7]])
        feature_counts = np.add(feature_counts, features)
    return feature_counts / len(trajectory)

def calculate_expected_value_difference(eval_policy, opt_policy, theta, rn = False):
    # if theta == "center":
    #     theta_vec = center_vec
    # elif theta == "anywhere":
    #     theta_vec = anywhere_vec
    # elif theta == "crash":
    #     theta_vec = crash_vec
    # else:
    #     theta_vec = hypotheses[theta]
    # theta_vec = theta_vec / np.linalg.norm(theta_vec)
    # V_eval = np.dot(theta_vec, expected_feature_counts(eval_policy))
    # V_opt = np.dot(theta_vec, expected_feature_counts(opt_policy))
    V_eval = reward(eval_policy, theta)
    V_opt = reward(opt_policy, theta)
    if rn:
        V_rand = 0
        for random_policy in random_policies:
            # V_rand += np.dot(theta_vec, expected_feature_counts(random_policy))
            V_rand += reward(random_policy, theta)
        V_rand /= len(random_policies)
        evd = (V_opt - V_eval) / (V_opt - V_rand + 0.0001)
    else:
        evd = V_opt - V_eval
    return evd

def calculate_policy_accuracy(opt_policy, eval_policy, epsilon = 0.0001):
    matches = 0
    compare_length = min(len(opt_policy), len(eval_policy))
    for i in range(compare_length):
        matches += abs(eval_policy[i][1] - opt_policy[i][1]) < epsilon
    return matches / len(opt_policy)

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

def generate_theta_vectors(policy, theta):
    theta_vecs = []
    initial_waypoint = policy[0]
    initial_state = initial_waypoint[3]
    initial_x = initial_state[0]
    for i, waypoint in enumerate(policy):
        state = waypoint[3]
        if theta == "center":
            theta_vec = center_vec
        elif theta == "anywhere":
            theta_vec = anywhere_vec
        elif theta == "crash":
            theta_vec = crash_vec
        theta_vec = theta_vec / np.linalg.norm(theta_vec)
        theta_vecs.append(theta_vec)
    return theta_vecs

def get_closest_theta(birl_sol):
    avg_similarities = []
    for pr in possible_rewards:
        for rp in random_policies:
            reward_theta_vecs = generate_theta_vectors(rp, pr)
            similarities = 0
            for rtv in reward_theta_vecs:
                cos_sim = np.dot(rtv, birl_sol) / (np.linalg.norm(rtv) * np.linalg.norm(birl_sol))
                similarities += np.abs(cos_sim)
        avg_similarities.append(similarities / len(reward_theta_vecs))
    return possible_rewards[np.argmax(avg_similarities)]
    # return "center"



"""
BIRL class
"""
class BIRL:
    def __init__(self, beta, epsilon = 0.0001):
        self.demos = {}
        self.demo_rewards = {theta: [] for theta in possible_rewards}
        self.counter_rewards = {theta: [] for theta in possible_rewards}
        self.counter_idx = 0
        self.epsilon = epsilon
        self.beta = beta
        self.num_mcmc_dims = num_features
    
    def birl(self, demos):
        ### Version A: new set of counterfactuals each time
        # self.demos = demos
        # probs = []
        # counters = []
        # print("ADDING TO CHOICE SET")
        # for theta in possible_rewards:
        #     counter = get_optimal_policy(theta, index = self.counter_idx)
        #     counters.append(counter)
        # for theta in possible_rewards:
        #     self.demo_rewards[theta].append(reward(self.demos[-1], theta))
        #     n = np.exp(self.beta * sum(self.demo_rewards[theta]))
        #     self.counter_rewards[theta].extend([reward(counter, theta) for counter in counters])
        #     d = sum(np.exp(self.beta * np.array(self.counter_rewards[theta]))) ** len(self.demos)
        #     probs.append(n/d)
        #     print("Likelihood for {}: demo score = {}, denominator = {}, probability = {}".format(theta, reward(self.demos[-1], theta), d, n/d))
        # Z = sum(probs)
        # pmf = np.asarray(probs) / Z
        # self.counter_idx += 1
        # return pmf, possible_rewards[np.argmax(pmf)], possible_policies[np.argmax(pmf)]

        ### Version B: same set of counterfactuals throughout
        probs = []
        for theta in possible_rewards:
            demo_reward = np.array([reward(demo, theta) for demo in demos], dtype = np.float32)
            counter_reward = np.array([reward(demo, theta) for demo in possible_policies], dtype = np.float32)
            n = np.exp(beta * sum(demo_reward))
            d = sum(np.exp(beta * counter_reward)) ** len(demos)
            probs.append(n/d)
        Z = sum(probs)
        pmf = np.asarray(probs) / Z
        return pmf, possible_rewards[np.argmax(pmf)], possible_policies[np.argmax(pmf)]

    def calc_ll(self, hyp_reward):
        hyp_reward = hyp_reward / np.linalg.norm(hyp_reward)
        ll = 1
        # num_demos = len(self.demonstrations)
        # choice_set = []
        # choice_set.extend(generate_optimal_demos(num_demos, "center"))
        # choice_set.extend(generate_optimal_demos(num_demos, "anywhere"))
        # choice_set.extend(generate_optimal_demos(num_demos, "crash"))
        # base_denominator = 0
        # for demo in choice_set:
        #     base_denominator += np.exp(self.beta * reward(demo, hyp_reward))
        for demo in self.demonstrations:
            numerator = np.exp(self.beta * reward(demo, hyp_reward))
            denominator = np.exp(self.beta * reward(demo, hyp_reward))
            for rand_demo in random_policies:
                denominator += np.exp(self.beta * reward(rand_demo, hyp_reward) / 3)
            ll *= numerator / denominator
        return ll

    def generate_proposal(self, old_sol, stdev, normalize):
        proposal_r = old_sol + stdev * np.random.randn(len(old_sol)) 
        if normalize:
            proposal_r = proposal_r / np.linalg.norm(proposal_r)
        return proposal_r

    def initial_solution(self):
        return np.zeros(self.num_mcmc_dims)  

    def run_mcmc(self, samples, stepsize, normalize=True, adaptive=False):
        num_samples = samples  # number of MCMC samples
        stdev = stepsize  # initial guess for standard deviation, doesn't matter too much
        accept_cnt = 0  # keep track of how often MCMC accepts

        # For adaptive step sizing
        accept_target = 0.4 # ideally around 40% of the steps accept; if accept count is too high, increase stdev, if too low reduce
        horizon = num_samples // 100 # how often to update stdev based on sliding window avg with window size horizon
        learning_rate = 0.05 # how much to update the stdev
        accept_cnt_list = [] # list of accepts and rejects for sliding window avg
        stdev_list = [] # list of standard deviations for debugging purposes
        accept_prob_list = [] # true cumulative accept probs for debugging purposes
        all_lls = [] # all the likelihoods

        self.chain = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        cur_sol = self.initial_solution() #initial guess for MCMC
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll
        map_sol = cur_sol

        for i in range(num_samples):
            # sample from proposal distribution
            prop_sol = self.generate_proposal(cur_sol, stdev, normalize)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
            all_lls.append(prop_ll)
            if prop_ll > cur_ll:
                # accept
                self.chain[i, :] = prop_sol
                accept_cnt += 1
                if adaptive:
                    accept_cnt_list.append(1)
                cur_sol = prop_sol
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_sol = prop_sol
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain[i, :] = prop_sol
                    accept_cnt += 1
                    if adaptive:
                        accept_cnt_list.append(1)
                    cur_sol = prop_sol
                    cur_ll = prop_ll
                else:
                    # reject
                    self.chain[i, :] = cur_sol
                    if adaptive:
                        accept_cnt_list.append(0)
            # Check for step size adaptation
            if adaptive:
                if len(accept_cnt_list) >= horizon:
                    accept_est = np.sum(accept_cnt_list[-horizon:]) / horizon
                    stdev = max(0.00001, stdev + learning_rate/np.sqrt(i + 1) * (accept_est - accept_target))
                stdev_list.append(stdev)
                accept_prob_list.append(accept_cnt / len(self.chain))
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol

    def get_map_solution(self):
        return self.map_sol

    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        burn_indx = int(len(self.chain) * burn_frac)
        mean_r = np.mean(self.chain[burn_indx::skip_rate], axis=0)
        return mean_r

"""
Demo sufficiency constants
"""
# possible_rewards = [key for key in main_hypotheses] + [key for key in alt_hypotheses if key not in failed_hypotheses]
possible_rewards = ["center", "anywhere", "crash"] + ["hypo67", "hypo55"]
possible_policies = [get_optimal_policy(theta) for theta in possible_rewards]
num_hypotheses = len(possible_rewards)