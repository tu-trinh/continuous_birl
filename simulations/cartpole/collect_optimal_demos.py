import gym
import cartpole_theta
import torch
import numpy as np
from train_optimal_agent import QNetwork
import pickle


def get_optimal_left(episodes):
    env = gym.make("CartpoleTheta-v0")
    dataset = []
    for episode in range(episodes):
        state = env.reset(theta="left")
        xi = []
        for t in range(500):
            action = 0
            xi.append([action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_optimal_right(episodes):
    env = gym.make("CartpoleTheta-v0")
    dataset = []
    for episode in range(episodes):
        state = env.reset(theta="right")
        xi = []
        for t in range(500):
            action = 1
            xi.append([action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_optimal_up(episodes):
    env = gym.make("CartpoleTheta-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("models/cartpole_up.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    for episode in range(episodes):
        state = env.reset(theta="up")
        xi = []
        for t in range(500):
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)
            xi.append([action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_optimal_tilt(episodes):
    env = gym.make("CartpoleTheta-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("models/cartpole_tilt.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    for episode in range(episodes):
        state = env.reset(theta="tilt")
        xi = []
        for t in range(500):
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)
            xi.append([action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset


def main():

    optimals = {'up': [], 'tilt': [], 'left': [], 'right': []}
    optimals['up'] = get_optimal_up(5)
    optimals['tilt'] = get_optimal_tilt(5)
    optimals['left'] = get_optimal_left(5)
    optimals['right'] = get_optimal_right(5)
    pickle.dump( optimals, open( "choices/optimal.pkl", "wb" ) )


if __name__ == "__main__":
    main()
