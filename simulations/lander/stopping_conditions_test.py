import random
import utils
from scipy.stats import norm
import numpy as np
import math
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = "nevd"

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # BIRL hyperparameters
    beta = 10.0 # confidence
    random_normalization = False # whether or not to normalize with random policy
    num_worlds = 1
    max_demos = 10
    true_theta = "center" # assume lander wants to land in center
    utils.generate_random_policies()

    # Experiment setup
    thresholds = [0.5] # thresholds on the a-VaR bounds

    for i in range(2):
        birl = utils.BIRL(beta)
        demo_sufficiency = False
        bounds = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        for M in range(max_demos):
            if demo_sufficiency:
                break
            if i == 0:
                D = utils.generate_bad_demos(M + 1, theta = true_theta)
            else:
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
                    demo_sufficiency = True

        # Output results for plotting
        if i == 0:
            print("bad demos")
        else:
            print("good demos")
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Policy loss bounds")
            for apl in bounds[threshold]:
                print(apl)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
        print("**************************************************")