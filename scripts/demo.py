# Script to run env with a particular q network
import gym
import LunarLander_c1
import torch
from train import QNetwork
import numpy as np

if __name__ == "__main__":
    env = gym.make('LunarLanderC1-v0')
    env.seed(0)
    reward_type = 1
    episodes = 25
    model_name = 'dqn_R' + str(reward_type) + '.pth'
    # model_name = 'dqn.pth'
    path = "../models/" + model_name
    qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
    qnetwork.load_state_dict(torch.load(path))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)

    for episode in range(episodes):
        state = env.reset(reward_type=reward_type)
        score = 0
        for t in range(1000):
            # get the best action according to q-network
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)

            # apply that action and take a step
            env.render()              # can always toggle visualization
            next_state, reward, done, info = env.step(action)
            score += reward
            print(reward)
            print(info['reward'])
            print(info['mod_reward'])
            print(info['reward_type'])
            if done:
                break
        # print(score)

    env.close()