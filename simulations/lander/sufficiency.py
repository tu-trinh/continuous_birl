import random
import utils
import numpy as np
import math
from scipy.stats import norm
import time

if __name__ == "__main__":
    start_time = time.time()
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    debug = False

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    beta = utils.beta
    random_normalization = True
    num_worlds = 1
    true_theta = "center"

    # Experiment setup
    policies = [utils.get_optimal_policy(true_theta) for _ in range(num_worlds)] # assume lander wants to land in center
    possible_rewards = utils.possible_rewards
    possible_policies = [utils.get_optimal_policy(theta) for theta in possible_rewards]
    demos = [[] for _ in range(num_worlds)]
    max_demos = 1
    utils.generate_random_policies()

    # Metrics to evaluate thresholds
    true_evds = {num_demos + 1: [] for num_demos in range(max_demos)}
    bounds = {num_demos + 1: 0 for num_demos in range(max_demos)} ## new
    pmfs = {num_demos + 1: 0 for num_demos in range(max_demos)}
    learned_policies = {num_demos + 1: 0 for num_demos in range(max_demos)}
    learned_rewards = {num_demos + 1: 0 for num_demos in range(max_demos)} ## new
    comparison_grids = {num_demos + 1: 0 for num_demos in range(max_demos)}

    for i in range(num_worlds):
        for M in range(max_demos):
            D = utils.generate_optimal_demos(M + 1)
            demos[i] = D

            birl = utils.BIRL(beta)
            map_pmf, map_sol, map_policy = birl.birl(demos[i])
            learned_policies[M + 1] = map_policy
            learned_rewards[M + 1] = map_sol
            pmfs[M + 1] = map_pmf

            #run counterfactual policy loss calculations using eval policy
            for j in range(len(possible_rewards)):
                pr = possible_rewards[j]
                Zi = utils.calculate_expected_value_difference(map_policy, possible_policies[j], pr, rn = random_normalization)
                true_evds[M + 1].append(Zi)
            
            N_burned = len(possible_rewards)
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
            if k >= N_burned:
                k = N_burned - 1
            bounds[M + 1] = sorted(true_evds[M + 1])[k]
            grid = utils.comparison_grid(possible_policies)
            comparison_grids[M + 1] = grid
    
    print("True reward function")
    print(true_theta)
    # print("True optimal policy")
    # print(utils.listify(policies[i]))
    # print("Possible policies")
    # poss_pols = []
    # for pp in possible_policies:
    #     poss_pols.append(utils.listify(pp))
    # print(poss_pols)
    for nd in range(max_demos):
        print("Num demos", nd + 1)
        print("Bounds")
        print(bounds[nd + 1])
        # print("EVDs")
        # print(true_evds[nd + 1])
        # print("PMFs")
        # print(list(pmfs[nd + 1]))
        # print("Learned policies")
        # print(utils.listify(learned_policies[nd + 1]))
        print("Learned rewards")
        print(learned_rewards[nd + 1])
        # print("Comparison grid")
        # print(utils.listify(comparison_grids[nd + 1], policy = False))
    print("**************************************************")

    end_time = time.time()
    print("Time took to run: {}".format((end_time - start_time) / 60))
