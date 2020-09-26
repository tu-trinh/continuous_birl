import numpy as np
import math
import pickle
import time
import copy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.interpolate import interp1d
from env import Task
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



def replay_demo(xi):
    T = 10.0
    traj = Trajectory(xi, T)
    env = Task()
    state = env.reset()
    count = 0
    max_count = 10001
    timesteps = np.linspace(0, T, max_count)
    xi = []
    while count < max_count:
        curr_time = timesteps[count]
        if count % 1000 == 0:
            xi.append(state["joint_position"][0:7].tolist())
        count += 1
        q_des = traj.get(curr_time)
        qdot = 10 * (q_des - state["joint_position"][0:7])
        next_state, reward, done, info = env.step(qdot)
        state = next_state
    env.close()
    return xi


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





def main():

    N_WAYPOINTS = 11
    N_RESCALE = 100
    N_NOISY = 100
    N_SPARSE = 100

    folder = "demos"
    demos = []
    for filename in os.listdir(folder):
        xi = pickle.load(open(folder + "/" + filename, "rb"))
        demos.append(np.asarray(xi))

    """get the original human demonstrations"""
    D = []
    for xi in demos:
        xi1 = rescale(xi, N_WAYPOINTS, 1)
        # replay_demo(xi1)
        D.append(rescale(xi, N_WAYPOINTS, 1))

    """get the rescaled human demonstrations"""
    R = []
    for episode in range(N_RESCALE):
        xi = np.random.choice(demos)
        ratio = np.random.random()
        R.append(rescale(xi, N_WAYPOINTS, ratio))

    """get the noisy human demonstrations"""
    N = []
    for episode in range(N_NOISY):
        xi = np.random.choice(demos)
        start = np.random.randint(0, len(xi))
        length = np.random.randint(30,70)
        tau = np.random.uniform([-0.05]*7, [0.05]*7)
        xi1 = deform(xi, start, length, tau)
        # replay_demo(xi1)
        N.append(rescale(xi1, N_WAYPOINTS, 1))

    """get the sparse human demonstrations"""
    S = []
    for episode in range(N_SPARSE):
        xi = np.random.choice(demos)
        target = np.random.randint(10,50)
        duration = target + np.random.randint(10,50)
        xi1 = hold(xi, target, duration)
        # replay_demo(xi1)
        S.append(rescale(xi1, N_WAYPOINTS, 1))

    pickle.dump( D, open( "choices/demos.pkl", "wb" ) )
    pickle.dump( R, open( "choices/rescaled.pkl", "wb" ) )
    pickle.dump( N, open( "choices/noisy.pkl", "wb" ) )
    pickle.dump( S, open( "choices/sparse.pkl", "wb" ) )

if __name__ == "__main__":
    main()
