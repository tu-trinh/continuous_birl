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
        self.n_waypoints = xi.shape[0]
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


class TrajOpt(object):

    def __init__(self, theta=None, n_waypoints=4, home=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])):
        """ set hyperparameters """
        self.theta = theta
        self.n_waypoints = n_waypoints
        self.n_joints = len(home)
        self.home = home
        """ create initial trajectory """
        self.xi0 = np.zeros((self.n_waypoints,self.n_joints))
        for idx in range(self.n_waypoints):
            self.xi0[idx,:] = self.home - idx * np.asarray([-0.1, -0.2, 0, 0, 0, 0, 0])
        self.xi0 = self.xi0.reshape(-1)
        """ create start point equality constraint """
        self.B = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))
        for idx in range(self.n_joints):
            self.B[idx,idx] = 1
        self.cons = LinearConstraint(self.B, self.home, self.home)
        self.gamma_mid = np.zeros((self.n_waypoints, 3))
        self.gamma_almost = np.zeros((self.n_waypoints, 3))
        self.gamma_end = np.zeros((self.n_waypoints, 3))

    """ forward kinematics of panda robot arm """
    def joint2pose(self, q):
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
    def trajcost(self, xi):
        # get trajectory in joint space and end-effector space
        xi = xi.reshape(self.n_waypoints,self.n_joints)
        for idx in range(self.n_waypoints):
            p_midway, p_almost, p_end = self.joint2pose(xi[idx,:])
            self.gamma_mid[idx,:] = p_midway
            self.gamma_almost[idx,:] = p_almost
            self.gamma_end[idx,:] = p_end
        # make trajectory smooth (in joint space)
        smoothcost_xi = 0
        for idx in range(1, self.n_waypoints):
            smoothcost_xi += np.linalg.norm(xi[idx,:] - xi[idx-1,:])**2
        # make trajectory avoid the vertical obstacle
        obscost = 0
        obsradius = 0.4
        obsposition = np.array([0.5, 0.0])
        for idx in range(1, self.n_waypoints):
            dist2obs_mid = np.linalg.norm(self.gamma_mid[idx,0:2] - obsposition)
            dist2obs_almost = np.linalg.norm(self.gamma_almost[idx,0:2] - obsposition)
            dist2obs_end = np.linalg.norm(self.gamma_end[idx,0:2] - obsposition)
            obscost += max([obsradius - dist2obs_mid, obsradius - dist2obs_almost, obsradius - dist2obs_end, 0])
        # make trajectory go to goal
        goalposition = np.array([0.75, 0.0, 0.1])
        goalcost = np.linalg.norm(self.gamma_end[-1,:] - goalposition)**2 * 10
        # make trajectory stay close to ground
        heightcost = 0
        for idx in range(1, self.n_waypoints):
            heightcost += self.gamma_end[idx,2]**2
        # weight each cost element
        return 0.2*smoothcost_xi + self.theta[0]*goalcost + \
            self.theta[1]*heightcost + self.theta[2]*obscost * 4

    """ use scipy optimizer to get optimal trajectory """
    def optimize(self, method='SLSQP'):
        start_t = time.time()
        res = minimize(self.trajcost, self.xi0, method=method, constraints=self.cons)
        xi = res.x.reshape(self.n_waypoints,self.n_joints)
        return xi, res, time.time() - start_t


def get_optimal(theta):
    opt = TrajOpt(theta=theta)
    xi, res, solve_time = opt.optimize()
    print("it took me this long to solve: ", solve_time)

    T = 5.0
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
    return [xi]


def main():

    optimals = {'all': [], 'goal': [], 'height': [], 'obs': []}
    optimals['all'] = get_optimal([1.0, 1.0, 1.0])
    optimals['goal'] = get_optimal([0.2, 1.0, 1.0])
    optimals['height'] = get_optimal([1.0, 0.5, 1.0])
    optimals['obs'] = get_optimal([1.0, 1.0, 0.0])
    pickle.dump( optimals, open( "choices/optimal.pkl", "wb" ) )


if __name__ == "__main__":
    main()
