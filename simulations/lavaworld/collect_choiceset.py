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

    episodes = 11
    THETA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    choiceset = []
    for episode in range(episodes):
        print(episode * 1.0 / episodes * 100)
        lava = np.asarray([np.random.random()*0.5 + 0.25, np.random.random()*0.5 + 0.25])
        for theta_star in THETA:
            xi = get_human(theta_star, lava, type="regular")
            choiceset.append((xi, lava))


    pickle.dump( choiceset, open( "choices/choiceset.pkl", "wb" ) )



if __name__ == "__main__":
    main()
