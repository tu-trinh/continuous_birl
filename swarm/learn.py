import numpy as np
import matplotlib.pyplot as plt
import pickle

# Reward functions
def Reward(xi, theta):
    R = 0
    if theta == "regular":
        goal_gain = 1
        obs_gain = 1
    elif theta == "goal":
        goal_gain = 1
        obs_gain = 0.0001
    elif theta == "obstacle":
        goal_gain = 0.0001
        obs_gain = 1
    for waypoint in xi:
        # info = waypoint[2]
        obs_distances = waypoint[1]
        # states = info["state"]
        states =waypoint[2]
        for i,state in enumerate(states):
            pos = state[0]
            cost = goal_gain * (pos[0]*pos[0] + pos[1]*pos[1]) +\
                    obs_gain * obs_distances[i]
            # print(cost)
            reward = .0001/cost
            R +=  reward
    return R

def get_belief(beta, D, Xi_R):
    rewards_D_1 = np.asarray([Reward(xi,"regular") for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,"regular") for xi in Xi_R], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,"goal") for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,"goal") for xi in Xi_R], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,"obstacle") for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,"obstacle") for xi in Xi_R], dtype = np.float32)

    # Reward for landing in middle
    n1 = np.exp(beta*sum(rewards_D_1))
    d1 = sum(np.exp(beta*rewards_XiR_1))**len(D)
    p1 = n1/d1
    print("n1: ",n1)
    print("p1: ", p1)
    # Reward for landing anywhere

    n2 = np.exp(beta*sum(rewards_D_2))
    d2 = sum(np.exp(beta*rewards_XiR_2))**len(D)
    p2 = n2/d2

    # Reward for crashing in middle
    n3 = np.exp(beta*sum(rewards_D_3))
    d3 = sum(np.exp(beta*rewards_XiR_3))**len(D)
    p3 = n3/d3

    Z = p1 + p2 + p3
    b = [p1/Z, p2/Z, p3/Z]
    return b

def birl_belief(beta, D, O):
    rewards_D_1 = np.asarray([Reward(xi,"regular") for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,"regular") for xi in O["regular"]], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,"goal") for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,"goal") for xi in O["goal"]], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,"obstacle") for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,"obstacle") for xi in O["obstacle"]], dtype = np.float32)

    # Reward for landing in middle
    n1 = np.exp(beta*sum(rewards_D_1))
    d1 = np.exp(beta*sum(rewards_XiR_1))
    p1 = n1/d1

    # Reward for landing anywhere
    n2 = np.exp(beta*sum(rewards_D_2))
    d2 = np.exp(beta*sum(rewards_XiR_2))
    p2 = n2/d2

    # Reward for crashing in the middle
    n3 = np.exp(beta*sum(rewards_D_3))
    d3 = np.exp(beta*sum(rewards_XiR_3))
    p3 = n3/d3

    Z = p1 + p2 + p3
    b = [p1/Z, p2/Z, p3/Z]
    return b


def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    E = pickle.load( open( "choices/counterfactual.pkl", "rb" ) )
    N = pickle.load( open( "choices/noisy.pkl", "rb" ) )
    O = pickle.load( open( "choices/optimal.pkl", "rb" ) )

    """ our approach, with counterfactuals """
    Xi_R = D + E
    for beta in [0.07, 0.1, 0.2]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(3), b)
        plt.title("Counterfactuals with beta: {}".format(beta) )
        plt.show()

    # """ UT approach, with noise """
    Xi_R = D + N
    for beta in [0.07, 0.1, 0.2]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(3), b)
        plt.title("Noisy with beta: {}".format(beta))
        plt.show()

    """ classic approach, with matching feature counts """
    for beta in [0.07, 0.1, 0.2]:
        b = birl_belief(beta, D, O)
        plt.bar(range(3), b)
        plt.title("Classic with beta: {}".format(beta))
        plt.show()

if __name__ == "__main__":
    main()
