import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import entropy

#reward function for keeping pole vertical
def Rup(xi):
    R = 0
    for waypoint in xi:
        angle = waypoint[3]
        if abs(abs(angle) - 0.0) < 0.1:
            R += 1
    return R / 200

#reward function for moving right (action = 1)
def Rright(xi):
    R = 0
    for waypoint in xi:
        action = waypoint[0]
        R = R + action
    return R / 200

#reward function for moving left (action = 0)
def Rleft(xi):
    R = 0
    for waypoint in xi:
        action = waypoint[0]
        R = R + 1 - action
    return R / 200

#reward function for keeping pole at an angle of pi/12
def Rtilt(xi):
    R = 0
    for waypoint in xi:
        angle = waypoint[3]
        if abs(abs(angle) - np.pi/12) < 0.1:
            R += 1
    return R / 200


#bayesian inference for each reward given demonstrations and choice set
def get_belief(beta, D, Xi_R):

    #reward for keeping pole vertical
    n1 = np.exp(beta*sum([Rup(xi) for xi in D]))
    d1 = sum([np.exp(beta*Rup(xi)) for xi in Xi_R])**len(D)
    p1 = n1/d1

    #reward for moving cart right
    n2 = np.exp(beta*sum([Rright(xi) for xi in D]))
    d2 = sum([np.exp(beta*Rright(xi)) for xi in Xi_R])**len(D)
    p2 = n2/d2

    #reward for moving cart left
    n3 = np.exp(beta*sum([Rleft(xi) for xi in D]))
    d3 = sum([np.exp(beta*Rleft(xi)) for xi in Xi_R])**len(D)
    p3 = n3/d3

    #reward for keeping pole at an angle
    n4 = np.exp(beta*sum([Rtilt(xi) for xi in D]))
    d4 = sum([np.exp(beta*Rtilt(xi)) for xi in Xi_R])**len(D)
    p4 = n4/d4

    #normalize and print belief
    Z = p1 + p2 + p3 + p4
    b = [p1/Z, p2/Z, p3/Z, p4/Z]
    return b


#comparison to optimal feature counts
def birl_belief(beta, D, O):

    #reward for keeping pole vertical
    n1 = np.exp(beta*sum([Rup(xi) for xi in D]))
    d1 = np.exp(beta*sum([Rup(xi) for xi in O["up"]]))
    p1 = n1/d1

    #reward for moving cart right
    n2 = np.exp(beta*sum([Rright(xi) for xi in D]))
    d2 = np.exp(beta*sum([Rright(xi) for xi in O["right"]]))
    p2 = n2/d2

    #reward for moving cart left
    n3 = np.exp(beta*sum([Rleft(xi) for xi in D]))
    d3 = np.exp(beta*sum([Rleft(xi) for xi in O["left"]]))
    p3 = n3/d3

    #reward for keeping pole at an angle
    n4 = np.exp(beta*sum([Rtilt(xi) for xi in D]))
    d4 = np.exp(beta*sum([Rtilt(xi) for xi in O["tilt"]]))
    p4 = n4/d4

    #normalize and print belief
    Z = p1 + p2 + p3 + p4
    b = [p1/Z, p2/Z, p3/Z, p4/Z]
    return b

def get_uncertainty(data):
    mean = np.mean(data)
    std = np.std(data)
    # print(len(data))
    sem = std/np.sqrt(len(data))
    return mean, sem

def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    E_set = pickle.load( open( "choices/counterfactuals_set.pkl", "rb" ) )
    N_set = pickle.load( open( "choices/noisies_set.pkl", "rb" ) )
    O = pickle.load( open( "choices/optimal.pkl", "rb" ) )
    # E = E_set[5]
    # N = N_set[5]
    # """ our approach, with counterfactuals """
    # Xi_R = D + E
    # for beta in [0.01, 0.1, 0.5]:
    #     b = get_belief(beta, D, Xi_R)
    #     plt.bar(range(4), b)
    #     plt.show()

    # """ UT approach, with noise """
    # Xi_R = D + N
    # for beta in [0.01, 0.1, 0.5]:
    #     b = get_belief(beta, D, Xi_R)
    #     plt.bar(range(4), b)
    #     plt.show()

    # """ classic approach, with matching feature counts """
    # for beta in [0.01, 0.1, 0.5]:
    #     b = birl_belief(beta, D, O)
    #     plt.bar(range(4), b)
    #     plt.show()

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
            belief_counter.append(b[0])
            entropy_counter.append(entropy(b))
            
            Xi_R = D + N
            b = get_belief(beta, D, Xi_R)
            belief_noise.append(b[0])
            entropy_noise.append(entropy(b))

            b = birl_belief(beta, D, O)
            belief_classic.append(b[0])
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
