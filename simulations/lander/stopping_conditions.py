import random
import utils
import copy
from scipy.stats import norm
import numpy as np
import math
import sys
import warnings
import time


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = sys.argv[1] # options: nevd, baseline, patience

    start_time = time.time()

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # BIRL hyperparameters
    beta = 10.0 # confidence
    # N = 20
    # step_stdev = 0.5
    # burn_rate = 0.05
    # skip_rate = 1
    random_normalization = True # whether or not to normalize with random policy
    # adaptive = True # whether or not to use adaptive step size
    num_worlds = 20
    max_demos = 10
    true_theta = "center" # assume lander wants to land in center
    utils.generate_random_policies()

    if stopping_condition == "nevd": # stop learning after passing a-VaR threshold
        # Experiment setup
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5] # thresholds on the a-VaR bounds

        # Metrics to evaluate thresholds
        bounds = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        pmfs = {threshold: [] for threshold in thresholds}
        learned_thetas = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true, P-NP

        for i in range(num_worlds):
            print("DOING WORLD", i + 1)
            birl = utils.BIRL(beta)
            for M in range(max_demos):
                print(M + 1, "demonstrations")
                D = utils.generate_optimal_demos(M + 1, theta = true_theta)
                if debug:
                    print("running BIRL with demos")
                    print("demos", D)
                map_pmf, map_sol = birl.birl(D)

                #run counterfactual policy loss calculations using eval policy
                policy_losses = []
                for j in range(utils.num_hypotheses):
                    Zj = utils.calculate_expected_value_difference(map_sol, utils.possible_rewards[j], rn = random_normalization) # compute policy loss
                    policy_losses.append((utils.possible_rewards[j], Zj))

                # compute VaR bound
                k = math.ceil(utils.num_hypotheses * alpha + norm.ppf(1 - delta) * np.sqrt(utils.num_hypotheses*alpha*(1 - alpha)) - 0.5)
                if k >= utils.num_hypotheses:
                    k = utils.num_hypotheses - 1
                policy_losses.sort(key = lambda pl : pl[1])
                avar_bound = policy_losses[k][1]
                print("policy losses", policy_losses)
                print("BOUND IS", avar_bound)

                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    actual = utils.calculate_expected_value_difference(map_sol, true_theta, rn = random_normalization)
                    # actual = 0
                    if avar_bound < threshold:
                        map_evd = actual
                        # store threshold metrics
                        bounds[threshold].append(avar_bound)
                        num_demos[threshold].append(M + 1)
                        pct_states[threshold].append((M + 1) / max_demos)
                        true_evds[threshold].append(map_evd)
                        avg_bound_errors[threshold].append(avar_bound - map_evd)
                        policy_optimalities[threshold].append(1)
                        policy_accuracies[threshold].append(utils.calculate_policy_accuracy(utils.possible_policies[utils.possible_rewards.index(true_theta)], utils.possible_policies[np.argmax(map_pmf)]))
                        confidence[threshold].add(i)
                        accuracies[threshold].append(avar_bound >= map_evd)
                        pmfs[threshold].append(map_pmf)
                        learned_thetas[threshold].append(map_sol)
                        if actual < threshold:
                            confusion_matrices[threshold][0][0] += 1
                        else:
                            confusion_matrices[threshold][0][1] += 1
                    else:
                        if actual < threshold:
                            confusion_matrices[threshold][1][0] += 1
                        else:
                            confusion_matrices[threshold][1][1] += 1

        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Policy loss bounds")
            for apl in bounds[threshold]:
                print(apl)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
            print("PMFs")
            for pmf in pmfs[threshold]:
                print(pmf)
            print("Learned reward functions")
            for lt in learned_thetas[threshold]:
                print(lt)
            print("Confidence")
            print(len(confidence[threshold]) / num_worlds)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            else:
                print(0.0)
            print("Confusion matrices")
            print(confusion_matrices[threshold])
        print("Reward hypotheses")
        for pr in utils.possible_rewards:
            print(pr, utils.all_hypotheses[pr])
        print("**************************************************")
    elif stopping_condition == "wfcb": # stop learning after avar bound < worst-case feature-count bound
        # Experiment setup
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate stopping condition
        accuracy = 0
        avg_bound_errors = []
        bounds = []
        true_evds = []
        num_demos = []
        pct_states = []
        policy_accuracies = []

        for i in range(num_worlds):
            env = envs[i]
            for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
                D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
                demos[i].append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.run_mcmc(N, step_stdev)
                #burn initial samples and skip every skip_rate for efficiency
                burn_indx = int(len(birl.chain) * burn_rate)
                samples = birl.chain[burn_indx::skip_rate]
                #check if MCMC seems to be mixing properly
                if debug:
                    print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
                    if birl.accept_rate > 0.7:
                        print("too high, probably need to increase standard deviation")
                    elif birl.accept_rate < 0.2:
                        print("too low, probably need to decrease standard dev")
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)
                #debugging to visualize the learned policy
                if debug:
                    print("map policy")
                    print("MAP weights", map_env.feature_weights)
                    # mdp_utils.visualize_policy(map_policy, env)
                    print("optimal policy")
                    print("true weights", env.feature_weights)
                    opt_policy = mdp_utils.get_optimal_policy(env)
                    # mdp_utils.visualize_policy(opt_policy, env)
                    policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                    print("policy accuracy", policy_accuracy)

                #run counterfactual policy loss calculations using eval policy
                policy_losses = []
                for sample in samples:
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(sample)
                    # learned_policy = mdp_utils.get_optimal_policy(learned_env)
                    Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization) # compute policy loss
                    policy_losses.append(Zi)

                # compute VaR bound
                N_burned = len(samples)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= len(policy_losses):
                    k = len(policy_losses) - 1
                policy_losses.sort()
                avar_bound = policy_losses[k]

                # compare bounds
                wfcb = mdp_utils.calculate_wfcb(map_policy, env, [demos[i]])
                print("WFCB", wfcb)
                if avar_bound < wfcb:
                    map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                    # store metrics
                    accuracy += avar_bound >= map_evd
                    avg_bound_errors.append(avar_bound - map_evd)
                    bounds.append(avar_bound)
                    true_evds.append(map_evd)
                    num_demos.append(M + 1)
                    pct_states.append((M + 1) / (num_rows * num_cols))
                    policy_accuracies.append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                    break
        
        # Output results for plotting
        print("Accuracy")
        print(accuracy / num_worlds)
        print("Bound errors")
        for abe in avg_bound_errors:
            print(abe)
        print("Policy loss bounds")
        for apl in bounds:
            print(apl)
        print("True EVDs")
        for tevd in true_evds:
            print(tevd)
        print("Num demos")
        for nd in num_demos:
            print(nd)
        print("Percent states")
        for ps in pct_states:
            print(ps)
        print("Policy accuracies")
        for pa in policy_accuracies:
            print(pa)
        print("**************************************************")
    elif stopping_condition == "wfcb_threshold": # stop learning after passing WFCB threshold
        # Experiment setup
        thresholds = [round(t, 2) for t in np.arange(start = 1.0, stop = -0.01, step = 0.79)] # thresholds on the WFCB bounds
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate thresholds
        accuracies = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        wfcb_bounds = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        confidence = {threshold: 0 for threshold in thresholds}

        for i in range(num_worlds):
            env = envs[i]
            start_comp = 0
            done_with_demos = False
            for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
                D = mdp_utils.generate_optimal_demo(env, demo_order[M])[:int(1/(1 - gamma))]
                demos[i].append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = bayesian_irl.BIRL(env, list(set([pair for traj in demos[i] for pair in traj])), beta) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.run_mcmc(N, step_stdev, adaptive = adaptive)
                #burn initial samples and skip every skip_rate for efficiency
                burn_indx = int(len(birl.chain) * burn_rate)
                samples = birl.chain[burn_indx::skip_rate]
                #check if MCMC seems to be mixing properly
                if debug:
                    print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
                    if birl.accept_rate > 0.7:
                        print("too high, probably need to increase standard deviation")
                    elif birl.accept_rate < 0.2:
                        print("too low, probably need to decrease standard dev")
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)
                #debugging to visualize the learned policy
                if debug:
                    print("map policy")
                    print("MAP weights", map_env.feature_weights)
                    # mdp_utils.visualize_policy(map_policy, env)
                    print("optimal policy")
                    print("true weights", env.feature_weights)
                    opt_policy = mdp_utils.get_optimal_policy(env)
                    # mdp_utils.visualize_policy(opt_policy, env)
                    policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                    print("policy accuracy", policy_accuracy)

                # compute WFCB bound
                wfcb = mdp_utils.calculate_wfcb(map_policy, env, demos[i])

                # evaluate thresholds
                for t in range(len(thresholds[start_comp:])):
                    threshold = thresholds[t + start_comp]
                    if wfcb < threshold:
                        map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                        # store threshold metrics
                        accuracies[threshold].append(wfcb >= map_evd)
                        avg_bound_errors[threshold].append(wfcb - map_evd)
                        wfcb_bounds[threshold].append(wfcb)
                        true_evds[threshold].append(map_evd)
                        num_demos[threshold].append(M + 1)
                        pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                        policy_accuracies[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                        confidence[threshold] += 1
                        if threshold == min(thresholds):
                            done_with_demos = True
                    else:
                        start_comp += t
                        break
                if done_with_demos:
                    break
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            print("Confidence")
            print(confidence[threshold] / num_worlds)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("WFCBs")
            for wfcb in wfcb_bounds[threshold]:
                print(wfcb)
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd * int(1/(1 - gamma)))
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps * int(1/(1 - gamma)))
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
        print("**************************************************")
    elif stopping_condition == "map_pi": # stop learning if additional demo does not change current learned policy
        # Experiment setup
        thresholds = [1, 2, 3, 4, 5]
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        elif world == "goal":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features, terminals = [random.randint(0, num_rows * num_cols - 1)]) for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate stopping condition
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}

        for i in range(num_worlds):
            env = envs[i]
            curr_map_pi = [-1 for _ in range(num_rows * num_cols)]
            patience = 0
            start_comp = 0
            done_with_demos = False
            for M in range(len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
                if demo_type == "pairs":
                    try:
                        D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
                        demos[i].append(D)
                    except IndexError:
                        pass
                    if debug:
                        print("running BIRL with demos")
                        print("demos", demos[i])
                    birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
                elif demo_type == "trajectories":
                    D = mdp_utils.generate_optimal_demo(env, demo_order[M])[:int(1/(1 - gamma))]
                    demos[i].append(D)
                    if debug:
                        print("running BIRL with demos")
                        print("demos", demos[i])
                    birl = bayesian_irl.BIRL(env, list(set([pair for traj in demos[i] for pair in traj])), beta)
                # use MCMC to generate sequence of sampled rewards
                birl.run_mcmc(N, step_stdev, adaptive = adaptive)
                #burn initial samples and skip every skip_rate for efficiency
                burn_indx = int(len(birl.chain) * burn_rate)
                samples = birl.chain[burn_indx::skip_rate]
                #check if MCMC seems to be mixing properly
                if debug:
                    print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
                    if birl.accept_rate > 0.7:
                        print("too high, probably need to increase standard deviation")
                    elif birl.accept_rate < 0.2:
                        print("too low, probably need to decrease standard dev")
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)
                #debugging to visualize the learned policy
                if debug:
                    print("map policy")
                    print("MAP weights", map_env.feature_weights)
                    # mdp_utils.visualize_policy(map_policy, env)
                    print("optimal policy")
                    print("true weights", env.feature_weights)
                    opt_policy = mdp_utils.get_optimal_policy(env)
                    # mdp_utils.visualize_policy(opt_policy, env)
                    policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                    print("policy accuracy", policy_accuracy)

                # compare policies
                policy_match = mdp_utils.calculate_policy_accuracy(curr_map_pi, map_policy)
                if policy_match == 1.0:
                    patience += 1
                    # evaluate thresholds
                    for t in range(len(thresholds[start_comp:])):
                        threshold = thresholds[t + start_comp]
                        if patience == threshold:
                            # store metrics
                            num_demos[threshold].append(M + 1)
                            if demo_type == "pairs":
                                pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                            elif demo_type == "trajectories":
                                pct_states[threshold].append((M + 1) * int(1/(1 - gamma)) / (num_rows * num_cols))
                            optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                            policy_optimalities[threshold].append(optimality)
                            policy_accuracies[threshold].append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                            confidence[threshold].add(i)
                            accuracies[threshold].append(optimality >= 0.96)
                            curr_map_pi = map_policy
                            if threshold == max(thresholds):
                                done_with_demos = True
                        else:
                            start_comp += t
                            break
                else:
                    patience = 0
                    curr_map_pi = map_policy
                if done_with_demos:
                    break
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
            print("Confidence")
            print(len(confidence[threshold]) / num_worlds)
            print("Accuracy")
            print(sum(accuracies[threshold]) / num_worlds)
            print("**************************************************")
    elif stopping_condition == "baseline": # stop learning once learned policy is some degree better than baseline policy
        # Experiment setup
        # thresholds = [round(t, 1) for t in np.arange(start = 0.0, stop = 1.1, step = 0.1)] # thresholds on the percent improvement
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Metrics to evaluate thresholds
        pct_improvements = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        pmfs = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true, P-NP

        for i in range(num_worlds):
            print("DOING WORLD", i + 1)
            birl = utils.BIRL(beta)
            base_theta = "hypo45"
            baseline_pi = utils.get_nonpessimal_policy(base_theta)
            baseline_evd = utils.calculate_expected_value_difference(base_theta, true_theta, rn = random_normalization)
            baseline_accuracy = utils.calculate_policy_accuracy(utils.possible_policies[utils.possible_rewards.index(true_theta)], baseline_pi)
            print("BASELINE POLICY: evd {} and policy accuracy {}".format(baseline_evd, baseline_accuracy))
            for M in range(max_demos):
                print(M + 1, "demonstrations")
                D = utils.generate_optimal_demos(M + 1, theta = true_theta)
                if debug:
                    print("running BIRL with demos")
                    print("demos", D)
                map_pmf, map_sol = birl.birl(D)

                # get percent improvements
                improvements = []
                for j in range(utils.num_hypotheses):
                    improvement = utils.calculate_percent_improvement(map_sol, utils.possible_rewards[j], base_theta)
                    improvements.append((utils.possible_rewards[j], improvement))

                # evaluate 95% confidence on lower bound of improvement
                k = math.ceil(utils.num_hypotheses*alpha + norm.ppf(1 - delta) * np.sqrt(utils.num_hypotheses*alpha*(1 - alpha)) - 0.5)
                if k >= utils.num_hypotheses:
                    k = utils.num_hypotheses - 1
                improvements.sort(key = lambda i : i[1], reverse = True)
                bound = improvements[k][1]
                print("percent improvements", improvements)
                print("BOUND IS", bound)
                
                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    actual = utils.calculate_percent_improvement(map_sol, true_theta, base_theta)
                    if bound > threshold:
                        map_evd = utils.calculate_expected_value_difference(map_sol, true_theta, rn = random_normalization)
                        # store threshold metrics
                        pct_improvements[threshold].append(bound)
                        num_demos[threshold].append(M + 1)
                        pct_states[threshold].append((M + 1) / max_demos)
                        true_evds[threshold].append(map_evd)
                        avg_bound_errors[threshold].append(actual - bound)
                        policy_optimalities[threshold].append(1)
                        policy_accuracies[threshold].append(utils.calculate_policy_accuracy(utils.possible_policies[utils.possible_rewards.index(true_theta)], utils.possible_policies[np.argmax(map_pmf)]))
                        confidence[threshold].add(i)
                        accuracies[threshold].append(bound <= actual)
                        if actual > threshold:
                            confusion_matrices[threshold][0][0] += 1
                        else:
                            confusion_matrices[threshold][0][1] += 1
                    else:
                        if actual > threshold:
                            confusion_matrices[threshold][1][0] += 1
                        else:
                            confusion_matrices[threshold][1][1] += 1
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Percent Improvements")
            for pi in pct_improvements[threshold]:
                print(pi)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
            print("Confidence")
            print(len(confidence[threshold]) / (num_worlds))
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            else:
                print(0.0)
            print("Confusion matrices")
            print(confusion_matrices[threshold])
        print("Reward hypotheses")
        for pr in utils.possible_rewards:
            print(pr, utils.all_hypotheses[pr])
        print("**************************************************")
    
    end_time = time.time()
    print("Time to run: {} minutes".format((end_time - start_time) / 60))
