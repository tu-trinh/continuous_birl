import random
import utils
import copy
import numpy as np

if __name__ == "__main__":
    env = utils.random_swarm()
    print(utils.get_optimal_policy(env))

    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    debug = False

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    beta = 10.0
    N = utils.N
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 1

    # Experiment setup
    envs = [utils.random_lander() for _ in range(num_worlds)]
    policies = [utils.get_optimal_policy(envs[i].feature_weights, envs[i].lava) for i in range(num_worlds)]
    # possible_rewards = [[0, 0.5], [0, 1], [0.5, 0], [0.5, 0.5], [0.5, 1], [1, 0], [1, 0.5], [1, 1]]
    possible_rewards = np.linspace(0, 1, N)
    demos = [[] for _ in range(num_worlds)]
    max_demos = 10
    utils.generate_random_policies()

    # Metrics to evaluate thresholds
    true_evds = {num_demos + 1: [] for num_demos in range(max_demos)}
    pmfs = {num_demos + 1: 0 for num_demos in range(max_demos)}
    learned_policies = {num_demos + 1: 0 for num_demos in range(max_demos)}
    comparison_grids = {num_demos + 1: 0 for num_demos in range(max_demos)}

    for i in range(num_worlds):
        env = envs[i]
        for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
            D = utils.generate_optimal_demo(env)
            demos[i].append(D)
            possible_policies = [utils.get_optimal_policy(pr, envs[i].lava) for pr in possible_rewards]
            birl = continuous_birl.CONT_BIRL(env, beta, possible_rewards, possible_policies)
            # use MCMC to generate sequence of sampled rewards
            birl.birl(demos[i])
            #generate evaluation policy from running BIRL
            map_env = copy.deepcopy(env)
            map_env.set_rewards(birl.get_map_solution())
            map_policy = birl.get_map_policy()
            learned_policies[M + 1] = map_policy
            map_pmf = birl.get_pmf()
            pmfs[M + 1] = map_pmf

            #run counterfactual policy loss calculations using eval policy
            for j in range(len(possible_rewards)):
                pr = possible_rewards[j]
                learned_env = copy.deepcopy(env)
                learned_env.set_rewards(pr)
                Zi = utils.calculate_expected_value_difference(map_policy, learned_env, possible_policies[j], rn = random_normalization) # compute policy loss
                true_evds[M + 1].append(Zi)
            
            grid = utils.comparison_grid(env, possible_rewards, possible_policies)
            comparison_grids[M + 1] = grid
    
    print("Environment")
    print(tuple(env.lava))
    print("True reward function")
    print(env.feature_weights)
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
        print(utils.listify(comparison_grids[nd + 1]))
    print("**************************************************")
