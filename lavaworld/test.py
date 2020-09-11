import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


"""Code used to collect demonstrations"""

def main():

    dataname = "easy"
    if dataname == "demo":
        hidden = 1
        stoptime_min = 200
        stoptime_max = 201
    if dataname == "easy":
        hidden = 1
        stoptime_min = 0
        stoptime_max = 50
    if dataname == "avoid":
        hidden = 0
        stoptime_min = 200
        stoptime_max = 201
    if dataname == "ignore":
        hidden = 0
        stoptime_min = 0
        stoptime_max = 1

    episodes = 50
    noise = 0.25
    step_length = 0.025
    hidden_radius = 0.3
    dataset = []

    for episode in range(episodes):
        state = np.asarray([0.0, 0.0])
        goal = np.asarray([1.0, 1.0])
        lava = np.asarray([random.random()*0.5 + 0.25, random.random()*0.5 + 0.25])
        dist2goal = np.linalg.norm(state - goal)
        dist2lava = np.linalg.norm(state - lava)
        stoptime = np.random.randint(stoptime_min, stoptime_max)
        xi, t = [], 0
        while True:

            action2goal = (goal - state) / dist2goal
            action2lava = (lava - state) / dist2lava

            if t >= stoptime:
                action = action2goal
            elif dist2lava > hidden_radius and hidden == True:
                action = action2goal
            else:
                action = action2goal - 0.75 * action2lava

            action += np.random.normal(0, noise, 2)
            action = step_length * action / np.linalg.norm(action)

            xi.append([t] +[action[0], action[1], state[0], state[1], dist2goal, dist2lava])
            state += action
            t += 1
            dist2goal = np.linalg.norm(state - goal)
            dist2lava = np.linalg.norm(state - lava)

            if dist2goal < 1e-1:
                dataset.append(xi)
                # uncomment to visualize
                # XI = np.asarray(xi)
                # plt.plot(XI[:,3], XI[:,4], 'o-')
                # plt.plot(goal[0], goal[1], 's')
                # plt.plot(lava[0], lava[1], 'x')
                # plt.show()
                break

    pickle.dump( dataset, open( dataname, "wb" ) )
    print(len(dataset))


if __name__ == "__main__":
    main()
