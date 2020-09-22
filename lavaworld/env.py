import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint



""" get optimal trajectory using constrained optimization """
def get_optimal(theta, lava_position):

    # hyperparameters
    n = 21
    vision_radius = 0.3

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
    res = minimize(trajcost, xi0, args=(theta, lava_position, n), method='SLSQP', constraints=cons)
    return res.x.reshape(n,2)
