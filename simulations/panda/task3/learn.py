import numpy as np
import matplotlib.pyplot as plt
import pickle


""" forward kinematics of panda robot arm """
def joint2pose(q):
    def RotX(q):
        return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
    def RotZ(q):
        return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def TransX(q, x, y, z):
        return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
    def TransZ(q, x, y, z):
        return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    H1 = TransZ(q[0], 0, 0, 0.333)
    H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
    H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
    H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
    H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
    H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
    H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
    H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
    H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
    return H[:,3][:3], H[:,0][:3], H[:,1][:3], H[:,2][:3]

""" problem specific cost function """
def trajcost(xi, theta):
    xi = np.asarray(xi)
    n_waypoints = len(xi)
    alignx = np.zeros((n_waypoints, 3))
    for idx in range(n_waypoints):
        p_end, align_x, align_y, align_z = joint2pose(xi[idx,:])
        alignx[idx,:] = align_x
    # feature counts
    pours = 0
    straights = 0
    for idx in range(1, n_waypoints):
        if abs(alignx[idx,0]) < 0.2:
            straights +=1
        else:
            pours +=1
    # weight each cost element
    return (1 - theta) * pours / 10. + theta * straights / 10.



BETA = [0.1, 0.5, 1.0]
THETA1 = 0.5
THETA2 = 0.25
THETA3 = 0.0
THETA = [THETA1, THETA2, THETA3]


#bayesian inference for each reward given demonstrations and choice set
def get_belief(beta, D, Xi_R):

    p = []
    for theta in THETA:
        n = np.exp(-beta*sum([trajcost(xi, theta) for xi in D]))
        d = sum([np.exp(-beta*trajcost(xi, theta)) for xi in Xi_R])**len(D)
        p.append(n/d)

    #normalize and print belief
    Z = sum(p)
    b = np.asarray(p) / Z
    return b

#comparison to optimal feature counts
def birl_belief(beta, D, O):

    avg_c1 = sum([trajcost(xi, THETA[0]) for xi in D]) / len(D)
    avg_c2 = sum([trajcost(xi, THETA[1]) for xi in D]) / len(D)
    avg_c3 = sum([trajcost(xi, THETA[2]) for xi in D]) / len(D)

    opt_c1 = trajcost(O["0.5"][0], THETA[0])
    opt_c2 = trajcost(O["0.25"][0], THETA[1])
    opt_c3 = trajcost(O["0.0"][0], THETA[2])

    p1 = np.exp(-beta*avg_c1)/np.exp(-beta*opt_c1)
    p2 = np.exp(-beta*avg_c2)/np.exp(-beta*opt_c2)
    p3 = np.exp(-beta*avg_c3)/np.exp(-beta*opt_c3)

    Z = p1 + p2 + p3
    b = [p1/Z, p2/Z, p3/Z]
    return b


def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    R = pickle.load( open( "choices/rescaled.pkl", "rb" ) )
    N = pickle.load( open( "choices/noisy.pkl", "rb" ) )
    S = pickle.load( open( "choices/sparse.pkl", "rb" ) )
    O = pickle.load( open( "choices/optimal.pkl", "rb" ) )

    """ our approach, with rescaled """
    Xi_R = D + R
    for beta in BETA:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(len(b)), b)
        plt.show()

    """ our approach, with noise """
    Xi_R = D + N
    for beta in BETA:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(len(b)), b)
        plt.show()

    """ our approach, with sparse """
    Xi_R = D + S
    for beta in BETA:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(len(b)), b)
        plt.show()

    """ our approach, all """
    Xi_R = D + R + N + S
    for beta in BETA:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(len(b)), b)
        plt.show()

    """ classic approach, with matching feature counts """
    for beta in BETA:
        b = birl_belief(beta, D, O)
        plt.bar(range(len(b)), b)
        plt.show()


if __name__ == "__main__":
    main()
