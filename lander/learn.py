import numpy as np
import matplotlib.pyplot as plt
import pickle

# Reward functions
def Reward(xi, theta):
    R = 0
    shaping = 0
    prev_shaping = None
    reward = 0
    initial_waypoint = xi[0]
    initial_state = initial_waypoint[3]
    initial_x = initial_state[0]
    for i, waypoint in enumerate(xi):
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
    return R

def get_belief(beta, D, Xi_R):
    rewards_D_1 = np.asarray([Reward(xi,"center") for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,"center") for xi in Xi_R], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,"anywhere") for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,"anywhere") for xi in Xi_R], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,"crash") for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,"crash") for xi in Xi_R], dtype = np.float32)

    # Reward for landing in middle
    n1 = np.exp(beta*sum(rewards_D_1))
    d1 = sum(np.exp(beta*rewards_XiR_1))**len(D)
    p1 = n1/d1

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
    rewards_D_1 = np.asarray([Reward(xi,"center") for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,"center") for xi in O["center"]], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,"anywhere") for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,"anywhere") for xi in O["anywhere"]], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,"crash") for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,"crash") for xi in O["crash"]], dtype = np.float32)

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
    for beta in [0.001, 0.002, 0.005]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(3), b)
        plt.show()

    """ UT approach, with noise """
    Xi_R = D + N
    for beta in [0.001, 0.002, 0.005]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(3), b)
        plt.show()

    """ classic approach, with matching feature counts """
    for beta in [0.001, 0.002, 0.005]:
        b = birl_belief(beta, D, O)
        plt.bar(range(3), b)
        plt.show()

if __name__ == "__main__":
    main()
