import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import LunarLander_c1
import pickle
import random

def get_model_name(r_type):
    # R1 with no time stop 
    model_name = "dqn_R1.pth"
    # R2 with no time stops (counter factuals for R2)
    if r_type == 2:
        model_name = "dqn_R2.pth"
    # R3 with no time stops (counter factuals for R3)
    elif r_type == 3:
        model_name = 'dqn_R3.pth'

    return model_name

def generate_data(r_type, ep):
    
    model_name = get_model_name(r_type)
    env = gym.make('LunarLanderC1-v0')

    # load our trained q-network
    path = "../models/" + model_name
    qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
    qnetwork.load_state_dict(torch.load(path))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)

    # we'll rollout over N episodes
    episodes = ep
    score = 0

    for episode in range(episodes):   
        # reset to start
        state = env.reset(reward_type=r_type)
        episode_reward = 0

        for t in range(1000):
        # get the best action according to q-network
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)

            # apply that action and take a step
            env.render()              # can always toggle visualization
            next_state, _, done, info = env.step(action)
            reward = info['reward']
            awake = info['awake']
            state = next_state
            score += reward
            episode_reward += reward

            if done:
                break
    env.close()

def main():
    eps = 10
    i = 3
    generate_data(i, eps)    

if __name__ == "__main__":
    main()

