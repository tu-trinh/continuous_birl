import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import pickle
import random


"""Code used to collect demonstrations using trained Q-values"""

def main():

    # load environment
    env = gym.make("CartPole-v0")

    # load our trained q-network
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("../models/cartpoleUP.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)

    episodes = 20
    t_delay = 8
    dataname = 'demo.pkl'
    dataset = []

    for episode in range(episodes):
        state = env.reset()
        xi = []
        stoptime = random.randint(200, 200)
        action = 0
        for t in range(500):

            # print(stoptime, t, action)

            if t < stoptime and t%t_delay == 0:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    action_values = qnetwork(state_t)
                    action_values = softmax(action_values).cpu().data.numpy()[0]
                action = np.argmax(action_values)
            # action = 1

            # env.render()              # can always toggle visualization
            xi.append([t] + [action] + list(state))
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break

    env.close()
    pickle.dump( dataset, open( dataname, "wb" ) )
    print(dataset)
    print(len(dataset))


if __name__ == "__main__":
    main()
