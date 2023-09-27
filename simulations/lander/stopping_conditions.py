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

    stopping_condition = sys.argv[1] # options: nevd, baseline, patience, held_out
    rand_norm = sys.argv[2] # options: true, false

    start_time = time.time()

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # BIRL hyperparameters
    beta = 10.0 # confidence
    optimality_threshold = 0.96
    # N = 20
    # step_stdev = 0.5
    # burn_rate = 0.05
    # skip_rate = 1
    random_normalization = True if rand_norm == "true" else False # whether or not to normalize with random policy
    # adaptive = True # whether or not to use adaptive step size
    num_worlds = 20
    max_demos = 25
    true_theta = "anywhere" # assume lander wants to land in center
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
    elif stopping_condition == "map_pi": # stop learning if additional demo does not change current learned policy
        # Experiment setup
        thresholds = [1, 2, 3, 4, 5]

        # Metrics to evaluate stopping condition
        num_demos = {threshold: [] for threshold in thresholds}
        # pct_states = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        # policy_accuracies = {threshold: [] for threshold in thresholds}
        # confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        # Predicted by true
        cm100 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 100%/99%
        cm95 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 95%/96%
        cm90 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 90%/92%
        cm5 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.5
        cm4 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.4
        cm3 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.3
        cm2 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.2
        cm1 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.1

        for i in range(num_worlds):
            # print("DOING WORLD", i + 1)
            curr_map_sol = None
            patience = 0
            birl = utils.BIRL(beta)
            for M in range(max_demos):
                # print(M + 1, "demonstrations")
                D = utils.generate_optimal_demos(M + 1, theta = true_theta)
                if debug:
                    print(f"running BIRL with {M + 1} demos", D)
                map_pmf, map_sol = birl.birl(D)
                if debug:
                    print("Current:", curr_map_sol)
                    print("Now:", map_sol)

                # compare policies
                if curr_map_sol == map_sol:
                    patience += 1
                    # evaluate thresholds
                    for t in range(len(thresholds)):
                        threshold = thresholds[t]
                        opt_policy_set = utils.main_policies[true_theta]
                        try:
                            eval_policy_set = utils.hypo_policies[map_sol]
                        except KeyError:
                            eval_policy_set = utils.main_policies[map_sol]
                        optimality = utils.calculate_policy_accuracy(opt_policy_set, eval_policy_set)
                        actual_nevd = utils.calculate_expected_value_difference(map_sol, true_theta, rn = random_normalization)
                        if patience == threshold:
                            # store metrics
                            num_demos[threshold].append(M + 1)
                            # optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                            policy_optimalities[threshold].append(optimality)
                            # policy_accuracies[threshold].append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                            # confidence[threshold].add(i)
                            accuracies[threshold].append(optimality >= optimality_threshold)
                            curr_map_sol = map_sol
                            # Evaluate actual positive by optimality
                            if optimality >= 1.0:
                                cm100[threshold][0][0] += 1
                            else:
                                cm100[threshold][0][1] += 1
                            if optimality >= 0.95:
                                cm95[threshold][0][0] += 1
                            else:
                                cm95[threshold][0][1] += 1
                            if optimality >= 0.90:
                                cm90[threshold][0][0] += 1
                            else:
                                cm90[threshold][0][1] += 1
                            # Evaluate actual positive by nEVD
                            if actual_nevd < 0.5:
                                cm5[threshold][0][0] += 1
                            else:
                                cm5[threshold][0][1] += 1
                            if actual_nevd < 0.4:
                                cm4[threshold][0][0] += 1
                            else:
                                cm4[threshold][0][1] += 1
                            if actual_nevd < 0.3:
                                cm3[threshold][0][0] += 1
                            else:
                                cm3[threshold][0][1] += 1
                            if actual_nevd < 0.2:
                                cm2[threshold][0][0] += 1
                            else:
                                cm2[threshold][0][1] += 1
                            if actual_nevd < 0.1:
                                cm1[threshold][0][0] += 1
                            else:
                                cm1[threshold][0][1] += 1
                        else:
                            # Evaluate actual positive by optimality
                            if optimality >= 1.0:
                                cm100[threshold][1][0] += 1
                            else:
                                cm100[threshold][1][1] += 1
                            if optimality >= 0.95:
                                cm95[threshold][1][0] += 1
                            else:
                                cm95[threshold][1][1] += 1
                            if optimality >= 0.90:
                                cm90[threshold][1][0] += 1
                            else:
                                cm90[threshold][1][1] += 1
                            # Evaluate actual positive by nEVD
                            if actual_nevd < 0.5:
                                cm5[threshold][1][0] += 1
                            else:
                                cm5[threshold][1][1] += 1
                            if actual_nevd < 0.4:
                                cm4[threshold][1][0] += 1
                            else:
                                cm4[threshold][1][1] += 1
                            if actual_nevd < 0.3:
                                cm3[threshold][1][0] += 1
                            else:
                                cm3[threshold][1][1] += 1
                            if actual_nevd < 0.2:
                                cm2[threshold][1][0] += 1
                            else:
                                cm2[threshold][1][1] += 1
                            if actual_nevd < 0.1:
                                cm1[threshold][1][0] += 1
                            else:
                                cm1[threshold][1][1] += 1
                else:
                    patience = 0
                    curr_map_sol = map_sol
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            # print("Percent states")
            # for ps in pct_states[threshold]:
            #     print(ps)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            # print("Policy accuracies")
            # for pa in policy_accuracies[threshold]:
            #     print(pa)
            # print("Confidence")
            # print(len(confidence[threshold]) / num_worlds)
            print("Accuracy")
            try:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            except ZeroDivisionError:
                print(0.0)
            print("CM100")
            print(cm100[threshold])
            print("CM95")
            print(cm95[threshold])
            print("CM90")
            print(cm90[threshold])
            print("CM5")
            print(cm5[threshold])
            print("CM4")
            print(cm4[threshold])
            print("CM3")
            print(cm3[threshold])
            print("CM2")
            print(cm2[threshold])
            print("CM1")
            print(cm1[threshold])
            print("**************************************************")
    elif stopping_condition == "baseline": # stop learning once learned policy is some degree better than baseline policy
        # Experiment setup
        thresholds = [round(t, 1) for t in np.arange(start = 0.0, stop = 1.1, step = 0.1)] # thresholds on the percent improvement
        # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    elif stopping_condition == "held_out":
        # Experiment setup
        thresholds = [3, 4, 5, 6, 7]  # every X demo, add it to the set

        # Metrics to evaluate thresholds
        num_demos = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        total_demos = {threshold: [] for threshold in thresholds}
        held_out_sets = {threshold: [] for threshold in thresholds}
        # Predicted by true
        cm100 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 100%/99%
        cm95 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 95%/96%
        cm90 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 90%/92%
        cm5 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.5
        cm4 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.4
        cm3 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.3
        cm2 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.2
        cm1 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.1

        for i in range(num_worlds):
            demo_counter = 0
            print("DOING WORLD", i + 1)
            birl = utils.BIRL(beta)
            for M in range(max_demos):
                # print(M + 1, "demonstrations")
                demo_counter += 1
                D = utils.generate_optimal_demos(M + 1, theta = true_theta)

                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    # print("Looking at threshold", threshold, "with demo counter at", demo_counter)
                    if demo_counter % threshold == 0:
                        held_out_sets[threshold].append(D[-1])
                        # print("Held out set length:", len(held_out_sets[threshold]))
                        continue
                    else:
                        total_demos[threshold].append(D[-1])
                        # print("Total demos length:", len(total_demos[threshold]))
                        map_pmf, map_sol = birl.birl(total_demos[threshold])
                        if len(held_out_sets[threshold]) >= 3:
                            done = map_sol == true_theta
                            opt_policy_set = utils.main_policies[true_theta]
                            try:
                                eval_policy_set = utils.hypo_policies[map_sol]
                            except KeyError:
                                eval_policy_set = utils.main_policies[map_sol]
                            optimality = utils.calculate_policy_accuracy(opt_policy_set, eval_policy_set)
                            actual_nevd = utils.calculate_expected_value_difference(map_sol, true_theta, rn = random_normalization)
                            if done:
                                # store threshold metrics
                                num_demos[threshold].append(M + 1)
                                policy_optimalities[threshold].append(optimality)
                                accuracies[threshold].append(optimality >= optimality_threshold)
                                # Evaluate actual positive by optimality
                                if optimality >= 1.0:
                                    cm100[threshold][0][0] += 1
                                else:
                                    cm100[threshold][0][1] += 1
                                if optimality >= 0.95:
                                    cm95[threshold][0][0] += 1
                                else:
                                    cm95[threshold][0][1] += 1
                                if optimality >= 0.90:
                                    cm90[threshold][0][0] += 1
                                else:
                                    cm90[threshold][0][1] += 1
                                # Evaluate actual positive by nEVD
                                if actual_nevd < 0.5:
                                    cm5[threshold][0][0] += 1
                                else:
                                    cm5[threshold][0][1] += 1
                                if actual_nevd < 0.4:
                                    cm4[threshold][0][0] += 1
                                else:
                                    cm4[threshold][0][1] += 1
                                if actual_nevd < 0.3:
                                    cm3[threshold][0][0] += 1
                                else:
                                    cm3[threshold][0][1] += 1
                                if actual_nevd < 0.2:
                                    cm2[threshold][0][0] += 1
                                else:
                                    cm2[threshold][0][1] += 1
                                if actual_nevd < 0.1:
                                    cm1[threshold][0][0] += 1
                                else:
                                    cm1[threshold][0][1] += 1
                            else:
                                # Evaluate actual positive by optimality
                                if optimality >= 1.0:
                                    cm100[threshold][1][0] += 1
                                else:
                                    cm100[threshold][1][1] += 1
                                if optimality >= 0.95:
                                    cm95[threshold][1][0] += 1
                                else:
                                    cm95[threshold][1][1] += 1
                                if optimality >= 0.90:
                                    cm90[threshold][1][0] += 1
                                else:
                                    cm90[threshold][1][1] += 1
                                # Evaluate actual positive by nEVD
                                if actual_nevd < 0.5:
                                    cm5[threshold][1][0] += 1
                                else:
                                    cm5[threshold][1][1] += 1
                                if actual_nevd < 0.4:
                                    cm4[threshold][1][0] += 1
                                else:
                                    cm4[threshold][1][1] += 1
                                if actual_nevd < 0.3:
                                    cm3[threshold][1][0] += 1
                                else:
                                    cm3[threshold][1][1] += 1
                                if actual_nevd < 0.2:
                                    cm2[threshold][1][0] += 1
                                else:
                                    cm2[threshold][1][1] += 1
                                if actual_nevd < 0.1:
                                    cm1[threshold][1][0] += 1
                                else:
                                    cm1[threshold][1][1] += 1
                                
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            else:
                print(0.0)
            print("CM100")
            print(cm100[threshold])
            print("CM95")
            print(cm95[threshold])
            print("CM90")
            print(cm90[threshold])
            print("CM5")
            print(cm5[threshold])
            print("CM4")
            print(cm4[threshold])
            print("CM3")
            print(cm3[threshold])
            print("CM2")
            print(cm2[threshold])
            print("CM1")
            print(cm1[threshold])
        # print("Reward hypotheses")
        # for pr in utils.possible_rewards:
        #     print(pr, utils.all_hypotheses[pr])
        print("**************************************************")
    
    end_time = time.time()
    print("Time to run: {} minutes".format((end_time - start_time) / 60))
