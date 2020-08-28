import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt


def main():

    # load environment
    env = gym.make('LunarLander-v2')

    # load our trained q-network
    qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
    qnetwork.load_state_dict(torch.load("dqn.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)

    # we'll rollout over N episodes
    episodes = 5
    scores = []

    for episode in range(episodes):

        # reset to start
        state = env.reset()
        score = 0
        state_x = []
        state_y = []

        for t in range(1000):

            if (t%10 == 0) :
            # get the best action according to q-network
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    action_values = qnetwork(state_t)
                    action_values = softmax(action_values).cpu().data.numpy()[0]
                action = np.argmax(action_values)

            # apply that action and take a step
            env.render()              # can always toggle visualization
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward
            state_x.append(state[0])
            state_y.append(state[1]) 
            if done:
                break

        scores.append(score)
        #print(score)
        plt.plot(state_x,state_y)
        plt.axis([-0.75, 0.75, 0, 1.6])
        

    # give some idea of how things went
    env.close()
    print(scores)
    print("mean score: ", np.mean(np.array(scores)))
    print("std score: ", np.std(np.array(scores)))
    plt.show()


if __name__ == "__main__":
    main()
