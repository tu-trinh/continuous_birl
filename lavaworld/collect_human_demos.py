import numpy as np
import pickle
from env import get_optimal


""" get human trajectory, where lava is not immediately visible """
def get_human(theta, lava, vision_radius=0.3, type="regular"):
    xi_star = get_optimal(theta, lava)
    n = xi_star.shape[0]
    if type == "regular":
        stoptime_lb = n - 1
        noise_variance = 0.01
    elif type == "noise":
        stoptime_lb = n - 1
        noise_variance = 0.05
    elif type == "counterfactual":
        stoptime_lb = 0
        noise_variance = 0.05
    detect, eta = False, 0.0
    xi = np.zeros((n,2))
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
        state += action + np.random.normal(0, noise_variance, 2)
        xi[idx,:] = state
    stoptime = np.random.randint(stoptime_lb, n)
    xi[stoptime:,0] = np.linspace(xi[stoptime,0], 1, n - stoptime)
    xi[stoptime:,1] = np.linspace(xi[stoptime,1], 1, n - stoptime)
    xi[0,:] = [0,0]
    xi[-1,:] = [1,1]
    return xi


def main():

    episodes = 10
    theta_star = 1.0

    N = 10
    noisies_set = []
    counterfactuals_set = []
    for i in range(0,N):
        demos = []
        counterfactuals = []
        noisies = []
        for episode in range(episodes):
            print(episode * 1.0 / episodes * 100)
            lava = np.asarray([np.random.random()*0.5 + 0.25, np.random.random()*0.5 + 0.25])
            xi = get_human(theta_star, lava, type="regular")
            demos.append((xi, lava))
            for k in range(10):
                xi1 = get_human(theta_star, lava, type="noise")
                noisies.append((xi1, lava))
            for k in range(10):
                xi1 = get_human(theta_star, lava, type="counterfactual")
                counterfactuals.append((xi1, lava))
        noisies_set.append(noisies)
        counterfactuals_set.append(counterfactuals)

    pickle.dump( demos, open( "choices/demos.pkl", "wb" ) )
    pickle.dump( noisies_set, open( "choices/noisies_set.pkl", "wb" ) )
    pickle.dump( counterfactuals_set, open( "choices/counterfactuals_set.pkl", "wb" ) )


if __name__ == "__main__":
    main()
