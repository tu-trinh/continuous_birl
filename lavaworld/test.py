import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


# hyperparameters
n = 21
vision_radius = 0.3


""" get optimal trajectory using constrained optimization """
def get_optimal(theta, lava):

    xi0 = np.zeros((n,2))
    xi0[:,0] = np.linspace(0, 1, n)
    xi0[:,1] = np.linspace(0, 1, n)
    xi0 = xi0.reshape(-1)
    B = np.zeros((4,n*2))
    B[0,0] = 1
    B[1,1] = 1
    B[2,-2] = 1
    B[3,-1] = 1

    def trajcost(xi, theta, lava, n):
        xi = xi.reshape(n,2)
        smoothcost = 0
        for idx in range(n-1):
            smoothcost += np.linalg.norm(xi[idx+1,:] - xi[idx,:])**2
        avoidcost = 0
        for idx in range(n):
            avoidcost -= np.linalg.norm(xi[idx,:] - lava) / n
        return smoothcost + theta * avoidcost

    cons = LinearConstraint(B, [0, 0, 1, 1], [0, 0, 1, 1])
    res = minimize(trajcost, xi0, args=(theta, lava, n), method='SLSQP', constraints=cons)
    return res.x.reshape(n,2)


""" get human trajectory, where lava is not immediately visible """
def get_human(theta, lava, noise):

    detect, eta = False, 0.0
    xi = np.zeros((n,2))
    xi_star = get_optimal(theta, lava)
    xi0 = np.zeros((n,2))
    xi0[:,0] = np.linspace(0, 1, n)
    xi0[:,1] = np.linspace(0, 1, n)
    state = xi[0,:]

    for idx in range(1,n):
        dist2lava = np.linalg.norm(state - lava)
        if dist2lava < vision_radius:
            detect = True
        if detect:
            eta += 0.1
            if eta > 1.0:
                eta = 1.0
        action = eta * (xi_star[idx,:] - state) + (1 - eta) * (xi0[idx,:] - state)
        state += action + np.random.normal(0, noise, 2)
        xi[idx,:] = state

    xi[0,:] = [0,0]
    xi[-1,:] = [1,1]
    return xi


""" our proposed method of getting 'easier' trajectories """
def get_counterfactual(xi):
    stoptime = np.random.randint(0, n)
    xi1 = copy.deepcopy(xi)
    xi1[stoptime:,0] = np.linspace(xi[stoptime,0], 1, n - stoptime)
    xi1[stoptime:,1] = np.linspace(xi[stoptime,1], 1, n - stoptime)
    return xi1


""" collect and record demonstrations """
def main():

    episodes = 20
    theta_star = 1.0
    THETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    demos = []
    counterfactuals = []
    noisies = []
    optimals = {}
    for theta in THETAS:
        optimals[str(theta)] = []

    for episode in range(episodes):
        print(episode * 1.0 / episodes * 100)
        lava = np.asarray([np.random.random()*0.5 + 0.25, np.random.random()*0.5 + 0.25])
        xi = get_human(theta_star, lava, 0.01)
        demos.append((xi, lava))
        for k in range(10):
            xi1 = get_counterfactual(xi)
            counterfactuals.append((xi1, lava))
        for k in range(10):
            xi1 = get_human(theta_star, lava, 0.05)
            noisies.append((xi1, lava))
        for theta in THETAS:
            xi_star = get_optimal(theta, lava)
            optimals[str(theta)].append((xi_star, lava))
    pickle.dump( demos, open( "demos", "wb" ) )
    pickle.dump( counterfactuals, open( "counterfactuals", "wb" ) )
    pickle.dump( noisies, open( "noisies", "wb" ) )
    pickle.dump( optimals, open( "optimals", "wb" ) )


if __name__ == "__main__":
    main()
