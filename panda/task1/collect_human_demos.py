import numpy as np
import pickle
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.interpolate import interp1d
from env import Task
import sys


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


def main():

    xi1 = pickle.load( open( "demos/1.pkl", "rb" ) )
    xi2 = pickle.load( open( "demos/2.pkl", "rb" ) )
    xi3 = pickle.load( open( "demos/3.pkl", "rb" ) )
    xi4 = pickle.load( open( "demos/4.pkl", "rb" ) )
    xi5 = pickle.load( open( "demos/5.pkl", "rb" ) )

    demos = []
    demos.append(replay_demo(xi1))
    demos.append(replay_demo(xi2))
    demos.append(replay_demo(xi3))
    demos.append(replay_demo(xi4))
    demos.append(replay_demo(xi5))
    pickle.dump( demos, open( "choices/demos.pkl", "wb" ) )



if __name__ == "__main__":
    main()
