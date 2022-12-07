import random
import utils
import numpy as np

if __name__ == "__main__":
    print(utils.generate_optimal_demos(1))
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
    demos = [[] for _ in range(num_worlds)]
    max_demos = 10
    utils.generate_random_policies()

    # Metrics to evaluate thresholds
    true_evds = {num_demos + 1: [] for num_demos in range(max_demos)}
    pmfs = {num_demos + 1: 0 for num_demos in range(max_demos)}
    learned_policies = {num_demos + 1: 0 for num_demos in range(max_demos)}
    comparison_grids = {num_demos + 1: 0 for num_demos in range(max_demos)}

    for i in range(num_worlds):
        for M in range(max_demos):
            D = utils.generate_optimal_demos(M + 1)
            demos[i] = D
            possible_policies = [utils.get_optimal_policy(theta) for theta in possible_rewards]

            map_pmf, map_sol, map_policy = utils.birl(demos[i])
            learned_policies[M + 1] = map_policy
            pmfs[M + 1] = map_pmf

            #run counterfactual policy loss calculations using eval policy
            for j in range(len(possible_rewards)):
                pr = possible_rewards[j]
                Zi = utils.calculate_expected_value_difference(map_policy, possible_policies[j], pr, rn = random_normalization)
                true_evds[M + 1].append(Zi)
            
            grid = utils.comparison_grid(possible_policies)
            comparison_grids[M + 1] = grid
    
    print("True reward function")
    print(true_theta)
    print("True optimal policy")
    print(utils.listify(policies[i]))
    print("Possible policies")
    poss_pols = []
    for pp in possible_policies:
        poss_pols.append(utils.listify(pp))
    print(poss_pols)
    for nd in range(max_demos):
        print("Num demos", nd + 1)
        print("EVDs")
        print(true_evds[nd + 1])
        print("PMFs")
        print(list(pmfs[nd + 1]))
        print("Learned policies")
        print(utils.listify(learned_policies[nd + 1]))
        print("Comparison grid")
        print(utils.listify(comparison_grids[nd + 1], policy = False))
    print("**************************************************")
