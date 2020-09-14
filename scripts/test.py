import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import LunarLander_c1
import pickle
import random


def main(r_type, ep):
    # R1 with no time stop
    if r_type == 1:
        action_stop = 100
        doc_name = 'R1'
    # R1 with time stops
    elif r_type == 2:
        action_stop = 1
        doc_name = 'R1_easy'
    # R2 with no time stops
    else:
        action_stop = 100
        doc_name = 'R2'

    t_delay = range(1,11)
    scores = [None] * len(t_delay)
    modified_scores = [None] * len(t_delay)
    landings = [None] * len(t_delay)
    

    for i in range(len(t_delay)):
        # load environment
        dataset = []
        env = gym.make('LunarLanderC1-v0')

        # load our trained q-network
        qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
        qnetwork.load_state_dict(torch.load("../models/dqn_R2.pth"))
        qnetwork.eval()
        softmax = torch.nn.Softmax(dim=1)

        # we'll rollout over N episodes
        episodes = ep
        score = 0
        modified_score = 0
        print("Starting lunar lander with action delay: ", t_delay[i])

        for episode in range(episodes):
            
            
            landing = 0
            
            if (episode % 50 == 0):
                print("On episode: ", episode)
            # reset to start
            xi = []
            state = env.reset()
            state_x = []
            state_y = []
            episode_reward = 0
            modified_episode_reward = 0
            stop_time = random.randint(action_stop,100)
            #print(stop_time)
            for t in range(1000):

                if t < stop_time or not(r_type%2==0) and (t%t_delay[i] == 0) :
                # get the best action according to q-network
                    with torch.no_grad():
                        state_t = torch.from_numpy(state).float().unsqueeze(0)
                        action_values = qnetwork(state_t)
                        action_values = softmax(action_values).cpu().data.numpy()[0]
                    action = np.argmax(action_values)
                elif t > stop_time:
                    action = 0

                # apply that action and take a step
                #env.render()              # can always toggle visualization
                next_state, _, done, info = env.step(action)
                reward = info['reward']
                modified_reward = info['mod_reward']
                awake = info['awake']
                xi.append([t] + [action] + [awake] + [state])

                prev_state = state
                state = next_state
                score += reward
                modified_score += modified_reward
                episode_reward += reward
                modified_episode_reward += modified_reward
                state_x.append(state[0])
                state_y.append(state[1])

                if done and t < 1000:
                    # print("Length of trajectory: ",len(xi))
                    print("Reward: ", episode_reward)
                    # print("Lander Status: ",awake)
                    dataset.append(xi)
                    if awake:
                        landing += 1
                    break
                elif done:
                    print('T = ' + str(t))
                    break


        scores[i] = score/episodes
        modified_scores[i] = modified_score/episodes
        landings[i] = landing
        print("Mean reward for completed iteration: ", scores[i])

        env.close()
        dataname = '../data/lander_' + doc_name + '_t_' + str(t_delay[i]) + '.pkl'
        pickle.dump( dataset, open( dataname, "wb" ) )
        #print(dataset)
        #print(dataset[0])
        #print(len(dataset))

if __name__ == "__main__":
    episodes = 25
    for i in range(1,4):
        main(i, episodes)
