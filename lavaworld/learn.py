import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import entropy


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
    THETA = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
    THETA = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for theta in THETA:
        n = np.exp(-beta*sum([R(xi, theta, lava) for xi, lava in D]))
        d = np.exp(-beta*sum([R(xi, theta, lava) for xi, lava in O[str(theta)]]))
        p.append(n/d)

    #normalize and print belief
    Z = sum(p)
    b = np.asarray(p) / Z
    return b

def get_uncertainty(data):
    mean = np.mean(data)
    std = np.std(data)
    # print(len(data))
    sem = std/np.sqrt(len(data))
    return mean, sem



def main():

    # #import trajectories (that could be choices)
    # D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    # E = pickle.load( open( "choices/counterfactuals_set.pkl", "rb" ) )
    # N = pickle.load( open( "choices/noisies_set.pkl", "rb" ) )
    # O = pickle.load( open( "choices/optimal.pkl", "rb" ) )
    #
    # """ our approach, with counterfactuals """
    # Xi_R = D + E[0]
    # for beta in [0, 0.1, 1, 2]:
    #     b = get_belief(beta, D, Xi_R)
    #     print(entropy(b))
    #     plt.bar(range(len(b)), b)
    #     plt.show()
    #
    # """ UT approach, with noise """
    # Xi_R = D + N[0]
    # for beta in [0, 0.1, 1, 2]:
    #     b = get_belief(beta, D, Xi_R)
    #     plt.bar(range(len(b)), b)
    #     print(entropy(b))
    #     plt.show()
    #
    # """ classic approach, with matching feature counts """
    # for beta in [0, 0.1, 1, 2]:
    #     b = birl_belief(beta, D, O)
    #     plt.bar(range(len(b)), b)
    #     print(entropy(b))
    #     plt.show()


    D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    E_set = pickle.load( open( "choices/counterfactuals_set.pkl", "rb" ) )
    N_set = pickle.load( open( "choices/noisies_set.pkl", "rb" ) )
    O = pickle.load( open( "choices/optimal.pkl", "rb" ) )

    e_mean_classic = []
    e_sem_classic = []
    e_mean_counter = []
    e_sem_counter = []
    e_mean_noise = []
    e_sem_noise = []
    b_mean_classic = []
    b_sem_classic = []
    b_mean_counter = []
    b_sem_counter = []
    b_mean_noise = []
    b_sem_noise = []

    BETA = range(0,10)
    BETA = [b * 0.1 for b in BETA]
    # BETA = [0.001, 0.005]

    for beta in BETA:
        entropy_counter = []
        entropy_noise = []
        entropy_classic = []
        belief_counter = []
        belief_noise = []
        belief_classic = []
        for i in range(len(E_set)):
            E = E_set[i]
            N = N_set[i]

            Xi_R = D + E
            b = get_belief(beta, D, Xi_R)
            belief_counter.append(b[-1])
            entropy_counter.append(entropy(b))

            Xi_R = D + N
            b = get_belief(beta, D, Xi_R)
            belief_noise.append(b[-1])
            entropy_noise.append(entropy(b))

            b = birl_belief(beta, D, O)
            belief_classic.append(b[-1])
            entropy_classic.append(entropy(b))

        mean, sem = get_uncertainty(entropy_classic)
        e_mean_classic.append(mean)
        e_sem_classic.append(sem)

        mean, sem = get_uncertainty(entropy_counter)
        e_mean_counter.append(mean)
        e_sem_counter.append(sem)

        mean, sem = get_uncertainty(entropy_noise)
        e_mean_noise.append(mean)
        e_sem_noise.append(sem)

        mean, sem = get_uncertainty(belief_classic)
        b_mean_classic.append(mean)
        b_sem_classic.append(sem)

        mean, sem = get_uncertainty(belief_counter)
        b_mean_counter.append(mean)
        b_sem_counter.append(sem)

        mean, sem = get_uncertainty(belief_noise)
        b_mean_noise.append(mean)
        b_sem_noise.append(sem)
        print("Completed beta: ",beta)

    b_mean = [b_mean_classic, b_mean_noise, b_mean_counter]
    b_sem = [b_sem_classic, b_sem_noise, b_sem_counter]
    e_mean = [e_mean_classic, e_mean_noise, e_mean_counter]
    e_sem = [e_sem_classic, e_sem_noise, e_sem_counter]

    print(b_mean)

    beliefs = {"mean": b_mean, "sem": b_sem}
    entropies = {"mean": e_mean, "sem": e_sem}

    pickle.dump(beliefs, open( "choices/beliefs.pkl", "wb" ) )
    pickle.dump(entropies, open( "choices/entropies.pkl", "wb" ) )

if __name__ == "__main__":
    main()
