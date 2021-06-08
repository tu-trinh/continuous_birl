import numpy as np
import pickle
from env import get_optimal


""" collect and record demonstrations """
def main():

    THETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    episodes = 20
    optimals = {}
    for theta in THETAS:
        optimals[str(theta)] = []
    for episode in range(episodes):
        print(episode * 1.0 / episodes * 100)    
        lava = np.asarray([np.random.random()*0.5 + 0.25, np.random.random()*0.5 + 0.25])
        for theta in THETAS:
            xi_star = get_optimal(theta, lava)
            optimals[str(theta)].append((xi_star, lava))
    pickle.dump( optimals, open( "choices/optimal.pkl", "wb" ) )


if __name__ == "__main__":
    main()
