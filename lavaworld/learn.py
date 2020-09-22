import numpy as np
import matplotlib.pyplot as plt
import pickle


#parameterized reward function
def R(xi, theta, lava):
    n = xi.shape[0]
    smoothcost = 0
    for idx in range(n-1):
        smoothcost += np.linalg.norm(xi[idx+1,:] - xi[idx,:])**2
    avoidcost = 0
    for idx in range(n):
        avoidcost -= np.linalg.norm(xi[idx,:] - lava) / n
    return smoothcost + theta * avoidcost


#bayesian inference for each reward given demonstrations and choice set
def get_belief(beta, D, Xi_R):

    p = []
    THETA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for theta in THETA:
        n = np.exp(-beta*sum([R(xi, theta, lava) for xi, lava in D]))
        d = sum([np.exp(-beta*R(xi, theta, lava)) for xi, lava in Xi_R])**len(D)
        p.append(n/d)

    #normalize and print belief
    Z = sum(p)
    b = np.asarray(p) / Z
    return b

#comparison to optimal feature counts
def birl_belief(beta, D, O):

    p = []
    THETA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for theta in THETA:
        n = np.exp(-beta*sum([R(xi, theta, lava) for xi, lava in D]))
        d = np.exp(-beta*sum([R(xi, theta, lava) for xi, lava in O[str(theta)]]))
        p.append(n/d)

    #normalize and print belief
    Z = sum(p)
    b = np.asarray(p) / Z
    return b


def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    E = pickle.load( open( "choices/counterfactual.pkl", "rb" ) )
    N = pickle.load( open( "choices/noisy.pkl", "rb" ) )
    O = pickle.load( open( "choices/optimal.pkl", "rb" ) )

    """ our approach, with counterfactuals """
    Xi_R = D + E
    for beta in [0, 0.1, 1, 2]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(11), b)
        plt.show()

    """ UT approach, with noise """
    Xi_R = D + N
    for beta in [0, 0.1, 1, 2]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(11), b)
        plt.show()

    """ classic approach, with matching feature counts """
    for beta in [0, 0.1, 1, 2]:
        b = birl_belief(beta, D, O)
        plt.bar(range(11), b)
        plt.show()


if __name__ == "__main__":
    main()
