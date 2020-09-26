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
    H_mid = np.linalg.multi_dot([H1, H2, H3, H4])
    H_almost = np.linalg.multi_dot([H_mid, H5, H6])
    H_end = np.linalg.multi_dot([H_almost, H7, H_panda_hand])
    return H_mid[:,3][:3], H_almost[:,3][:3], H_end[:,3][:3]

""" problem specific cost function """
def trajcost(xi, theta):
    # get trajectory in joint space and end-effector space
    xi = np.asarray(xi)
    n_waypoints = len(xi)
    n_joints = 7
    gamma_mid = np.zeros((n_waypoints, 3))
    gamma_almost = np.zeros((n_waypoints, 3))
    gamma_end = np.zeros((n_waypoints, 3))
    for idx in range(n_waypoints):
        p_midway, p_almost, p_end = joint2pose(xi[idx,:])
        gamma_mid[idx,:] = p_midway
        gamma_almost[idx,:] = p_almost
        gamma_end[idx,:] = p_end
    # make trajectory smooth (in joint space)
    smoothcost_xi = 0
    for idx in range(1, n_waypoints):
        smoothcost_xi += np.linalg.norm(xi[idx,:] - xi[idx-1,:])**2
    # make trajectory avoid the vertical obstacle
    obscost = 0
    obsradius = 0.4
    obsposition = np.array([0.5, 0.0])
    for idx in range(1, n_waypoints):
        dist2obs_mid = np.linalg.norm(gamma_mid[idx,0:2] - obsposition)
        dist2obs_almost = np.linalg.norm(gamma_almost[idx,0:2] - obsposition)
        dist2obs_end = np.linalg.norm(gamma_end[idx,0:2] - obsposition)
        obscost += max([obsradius - dist2obs_mid, obsradius - dist2obs_almost, obsradius - dist2obs_end, 0])
    # make trajectory go to goal
    goalposition = np.array([0.75, 0.0, 0.1])
    goalcost = np.linalg.norm(gamma_end[-1,:] - goalposition)**2 * 10
    # make trajectory stay close to ground
    heightcost = 0
    for idx in range(1, n_waypoints):
        heightcost += gamma_end[idx,2]**2
    # weight each cost element
    return 0.2*smoothcost_xi + theta[0]*goalcost + \
        theta[1]*heightcost + theta[2]*obscost * 4.0


#bayesian inference for each reward given demonstrations and choice set
def get_belief(beta, D, Xi_R):

    p = []
    THETA = [[1, 1, 1], [0.2, 1, 1], [1, 0.5, 1], [1, 1, 0]]
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

    n1 = np.exp(-beta*sum([trajcost(xi, [1, 1, 1]) for xi in D]))
    d1 = np.exp(-beta*sum([trajcost(xi, [1, 1, 1]) for xi in O["all"]]))
    p1 = n1/d1

    n2 = np.exp(-beta*sum([trajcost(xi, [0.2, 1, 1]) for xi in D]))
    d2 = np.exp(-beta*sum([trajcost(xi, [0.2, 1, 1]) for xi in O["goal"]]))
    p2 = n2/d2

    n3 = np.exp(-beta*sum([trajcost(xi, [1, 0.5, 1]) for xi in D]))
    d3 = np.exp(-beta*sum([trajcost(xi, [1, 0.5, 1]) for xi in O["height"]]))
    p3 = n3/d3

    n4 = np.exp(-beta*sum([trajcost(xi, [1, 1, 0]) for xi in D]))
    d4 = np.exp(-beta*sum([trajcost(xi, [1, 1, 0]) for xi in O["obs"]]))
    p4 = n4/d4

    #normalize and print belief
    Z = p1 + p2 + p3 + p4
    b = [p1/Z, p2/Z, p3/Z, p4/Z]
    return b


def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "choices/demos.pkl", "rb" ) )
    E = pickle.load( open( "choices/counterfactual.pkl", "rb" ) )
    N = pickle.load( open( "choices/noisy.pkl", "rb" ) )
    O = pickle.load( open( "choices/optimal.pkl", "rb" ) )

    """ our approach, with counterfactuals """
    Xi_R = D + E
    for beta in [0.01, 0.05, 0.1]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(4), b)
        plt.show()

    """ UT approach, with noise """
    Xi_R = D + N
    for beta in [0.01, 0.05, 0.1]:
        b = get_belief(beta, D, Xi_R)
        plt.bar(range(4), b)
        plt.show()

    """ classic approach, with matching feature counts """
    for beta in [0.01, 0.05, 0.1]:
        b = birl_belief(beta, D, O)
        plt.bar(range(4), b)
        plt.show()


if __name__ == "__main__":
    main()
