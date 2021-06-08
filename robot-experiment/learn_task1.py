import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
import time
import copy
import sys
import os


class Trajectory(object):

    def __init__(self, xi, T):
        """ create cublic interpolators between waypoints """
        self.xi = np.asarray(xi)
        self.T = T
        self.n_waypoints = self.xi.shape[0]
        timesteps = np.linspace(0, self.T, self.n_waypoints)
        self.f1 = interp1d(timesteps, self.xi[:,0], kind='cubic')
        self.f2 = interp1d(timesteps, self.xi[:,1], kind='cubic')
        self.f3 = interp1d(timesteps, self.xi[:,2], kind='cubic')
        self.f4 = interp1d(timesteps, self.xi[:,3], kind='cubic')
        self.f5 = interp1d(timesteps, self.xi[:,4], kind='cubic')
        self.f6 = interp1d(timesteps, self.xi[:,5], kind='cubic')
        self.f7 = interp1d(timesteps, self.xi[:,6], kind='cubic')

    def get(self, t):
        """ get interpolated position """
        if t < 0:
            q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0), self.f7(0)]
        elif t < self.T:
            q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t), self.f7(t)]
        else:
            q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T), self.f5(self.T), self.f6(self.T), self.f7(self.T)]
        return np.asarray(q)


def rescale(xi, n_waypoints, ratio):
    xi1 = copy.deepcopy(np.asarray(xi)[:, 0:7])
    total_waypoints = xi1.shape[0] * ratio
    index = np.linspace(0, total_waypoints, n_waypoints)
    index = np.around(index, decimals=0).astype(int)
    index[-1] = min([index[-1], xi1.shape[0]-1])
    return xi1[index,:]

def deform(xi, start, length, tau):
    xi1 = copy.deepcopy(np.asarray(xi)[:, 0:7])
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(A.T @ A)
    U = np.zeros(length)
    gamma = np.zeros((length, 7))
    for idx in range(7):
        U[0] = tau[idx]
        gamma[:,idx] = R @ U
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end,:] += gamma[0:end-start,:]
    return xi1

def hold(xi, target, duration):
    xi1 = copy.deepcopy(np.asarray(xi)[:, 0:7])
    state = xi1[0,:]
    count = 0
    while True:
        next_idx = min([count+target, xi.shape[0]-1])
        action = (xi1[next_idx,:] - state) / (next_idx - count + 1)
        for idx in range(duration):
            xi1[count,:] = state
            state = copy.deepcopy(state + action)
            count += 1
            if count == xi1.shape[0]:
                return xi1



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



BETA = [0.35]
THETA1 = [1, 1, 1]
# THETA2 = [0.2, 1, 1]
# THETA3 = [1, 0.1, 1]
THETA4 = [1, 1, 0]
# THETA = [THETA1, THETA2, THETA3, THETA4]
THETA = [THETA1, THETA4]


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
    # avg_c3 = sum([trajcost(xi, THETA[2]) for xi in D]) / len(D)
    # avg_c4 = sum([trajcost(xi, THETA[3]) for xi in D]) / len(D)

    opt_c1 = trajcost(O["all"][0], THETA[0])
    # opt_c2 = trajcost(O["goal"][0], THETA[1])
    # opt_c3 = trajcost(O["height"][0], THETA[2])
    # opt_c4 = trajcost(O["obs"][0], THETA[2])
    opt_c2 = trajcost(O["obs"][0], THETA[1])

    p1 = np.exp(-beta*avg_c1)/np.exp(-beta*opt_c1)
    p2 = np.exp(-beta*avg_c2)/np.exp(-beta*opt_c2)
    # p3 = np.exp(-beta*avg_c3)/np.exp(-beta*opt_c3)
    # p4 = np.exp(-beta*avg_c4)/np.exp(-beta*opt_c4)

    Z = p1 + p2# + p3 + p4
    b = [p1/Z, p2/Z]#, p3/Z, p4/Z]
    return b


def main():

    belief_counterfactual = []
    belief_noise = []
    belief_classic = []
    for usernumber in range(1,11):
        usernumber = str(usernumber)
        N_WAYPOINTS = 11
        N_RESCALE = 100
        N_NOISY = 100
        N_SPARSE = 100

        folder = "demos_task1/user"+usernumber
        demos = []
        for filename in os.listdir(folder):
            xi = pickle.load(open(folder + "/" + filename, "rb"))
            demos.append(np.asarray(xi))

        """get the original human demonstrations"""
        D = []
        for xi in demos:
            xi1 = rescale(xi, N_WAYPOINTS, 1)
            D.append(rescale(xi, N_WAYPOINTS, 1))

        """get the rescaled human demonstrations"""
        R = []
        for episode in range(N_RESCALE):
            xi = demos[np.random.randint(0, len(D))]
            ratio = np.random.random()
            R.append(rescale(xi, N_WAYPOINTS, ratio))

        """get the noisy human demonstrations"""
        N = []
        for episode in range(N_NOISY):
            xi = demos[np.random.randint(0, len(D))]
            start = np.random.randint(0, len(xi))
            length = np.random.randint(30,70)
            tau = np.random.uniform([-0.05]*7, [0.05]*7)
            xi1 = deform(xi, start, length, tau)
            N.append(rescale(xi1, N_WAYPOINTS, 1))

        """get the sparse human demonstrations"""
        S = []
        for episode in range(N_SPARSE):
            xi = demos[np.random.randint(0, len(D))]
            target = np.random.randint(100,200)
            duration = target + np.random.randint(0,1)
            xi1 = hold(xi, target, duration)
            S.append(rescale(xi1, N_WAYPOINTS, 1))

        savefolder = "choices_task1/user"+usernumber
        pickle.dump( D, open( savefolder + "/demos.pkl", "wb" ) )
        pickle.dump( R, open( savefolder + "/rescaled.pkl", "wb" ) )
        pickle.dump( N, open( savefolder + "/noisy.pkl", "wb" ) )
        pickle.dump( S, open( savefolder + "/sparse.pkl", "wb" ) )

        #import trajectories (that could be choices)
        D = pickle.load( open( savefolder + "/demos.pkl", "rb" ) )
        R = pickle.load( open( savefolder + "/rescaled.pkl", "rb" ) )
        N = pickle.load( open( savefolder + "/noisy.pkl", "rb" ) )
        S = pickle.load( open( savefolder + "/sparse.pkl", "rb" ) )
        O = pickle.load( open( "choices_task1/optimal.pkl", "rb" ) )

        """ our approach, with noise """
        Xi_R = D + N
        for beta in BETA:
            b = get_belief(beta, D, Xi_R)
            belief_noise.append(b)
            print("noise belief: ", b)

        """ our approach, all """
        Xi_R = D + R + N + S
        for beta in BETA:
            b = get_belief(beta, D, Xi_R)
            belief_counterfactual.append(b)
            print("all belief: ", b)

        """ classic approach, with matching feature counts """
        for beta in BETA:
            b = birl_belief(beta, D, O)
            belief_classic.append(b)
            print("classic belief: ", b)

    belief = {'classic': belief_classic, 'noise': belief_noise,\
                'counterfactual': belief_counterfactual}
    print(belief)
    pickle.dump(belief, open("results/belief_task1.pkl", "wb") )



if __name__ == "__main__":
    main()
