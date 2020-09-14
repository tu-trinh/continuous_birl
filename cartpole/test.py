import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import pickle


def get_optimal_left(episodes):
    env = gym.make("CartPole-v0")
    dataset = []
    for episode in range(episodes):
        state = env.reset()
        xi = []
        for t in range(500):
            action = 0
            xi.append([t] + [action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_optimal_right(episodes):
    env = gym.make("CartPole-v0")
    dataset = []
    for episode in range(episodes):
        state = env.reset()
        xi = []
        for t in range(500):
            action = 1
            xi.append([t] + [action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_optimal_up(episodes):
    env = gym.make("CartPole-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("../models/cartpoleUP.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    for episode in range(episodes):
        state = env.reset()
        xi = []
        for t in range(500):
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)
            xi.append([t] + [action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_optimal_tilt(episodes):
    env = gym.make("CartPole-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("../models/cartpoleTILT.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    for episode in range(episodes):
        state = env.reset()
        xi = []
        for t in range(500):
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)
            xi.append([t] + [action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset

def get_human(episodes, t_delay=8, noise=0.0, counterfactual=False):
    env = gym.make("CartPole-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("../models/cartpoleUP.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    if counterfactual:
        stoptime_lb = 0
    else:
        stoptime_lb = 200
    for episode in range(episodes):
        stoptime = np.random.randint(stoptime_lb, 201)
        state = env.reset()
        xi = []
        action = 0
        for t in range(500):
            if t < stoptime and t%t_delay == 0:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    action_values = qnetwork(state_t)
                    action_values = softmax(action_values).cpu().data.numpy()[0]
                action = np.argmax(action_values)
            if np.random.random() < noise:
                action = np.random.randint(0,2)
            xi.append([t] + [action] + list(state))
            # env.render()              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset


def main():

    demos = get_human(20, 8, 0.0, False)
    counterfactuals = get_human(100, 8, 0.0, True)
    noisies = get_human(100, 8, 0.05, False)
    optimals = {'up': [], 'tilt': [], 'left': [], 'right': []}
    optimals['up'] = get_optimal_up(20)
    optimals['tilt'] = get_optimal_tilt(20)
    optimals['left'] = get_optimal_left(20)
    optimals['right'] = get_optimal_right(20)

    pickle.dump( demos, open( "demos", "wb" ) )
    pickle.dump( counterfactuals, open( "counterfactuals", "wb" ) )
    pickle.dump( noisies, open( "noisies", "wb" ) )
    pickle.dump( optimals, open( "optimals", "wb" ) )


if __name__ == "__main__":
    main()
