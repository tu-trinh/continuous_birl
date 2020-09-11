import numpy as np
import matplotlib.pyplot as plt
import pickle


#parameterized reward function
def R(xi, theta):
    dist2goal, dist2lava = 0, 0
    for waypoint in xi:
        dist2goal += waypoint[5]
        dist2lava_t = waypoint[6]
        if dist2lava_t < 0.3:
            dist2lava += 1
    dist2goal /= 1.0 * len(xi)
    dist2lava /= 1.0 * len(xi)
    # print(dist2goal, dist2lava)
    R = (1 - theta) * dist2goal + theta * dist2lava
    return -1.0 * R


#bayesian inference for each reward given demonstrations and choice set
def get_belief(beta, D, Xi_R):

    p = []
    THETA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for theta in THETA:
        n = np.exp(beta*sum([R(xi, theta) for xi in D]))
        d = sum([np.exp(beta*R(xi, theta)) for xi in Xi_R])**len(D)
        p.append(n/d)

    #normalize and print belief
    Z = sum(p)
    b = np.asarray(p) / Z
    print("Belief: ", b)


def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "demo", "rb" ) )
    E = pickle.load( open( "easy", "rb" ) )
    Davoid = pickle.load( open( "avoid", "rb" ) )
    Dignore = pickle.load( open( "ignore", "rb" ) )

    #build choice set --- default includes demonstrations and easy simplifications
    Xi_R = D + E

    #optionally, you can also include optimal trajectories for each reward function.
    #if the human had no limitations, we would expect them to show one of these!
    # Xi_R += Davoid
    # Xi_R += Dignore

    #rationality constant. Increasing makes different terms dominate
    print(len(Xi_R))
    for beta in [0, 0.1, 1, 2]:
        get_belief(beta, D, Xi_R)




if __name__ == "__main__":
    main()
