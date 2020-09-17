import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import LunarLander_c1
import pickle
import random

def get_params(r_type,max_action_stop):
    model_name = "dqn_R1.pth"
    action_stop = max_action_stop
    # R1 with no time stop
    if r_type == 1:    
        doc_name = 'R1'
    # R1 with time stops
    elif r_type == 2:
        action_stop = 1
        doc_name = 'R1_easy'
    # R1 with noise
    elif r_type == 3:
        doc_name = 'R1_noisy'
    # R2 with no time stops (counter factuals for R2)
    elif r_type == 4:
        doc_name = 'R2'
        model_name = "dqn_R2.pth"
    else:
        doc_name = 'R3'
        model_name = 'dqn_R3.pth'


    return action_stop, doc_name, model_name

def generate_data(r_type, ep):
    max_action_stop = 1000
    action_stop, doc_name, model_name = get_params(r_type, max_action_stop)

    # Action delays. Decide how many frames between actions
    t_delay = range(1,11)
    scores = [None] * len(t_delay)
    modified_scores = [None] * len(t_delay)
    landings = [None] * len(t_delay)
    

    for delay in t_delay:
        # load environment
        dataset = []
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
        modified_score = 0
        print("Starting lunar lander with action delay: ", delay)

        for episode in range(episodes):   
            landing = 0
            
            # reset to start
            xi = []
            state = env.reset()
            episode_reward = 0
            
            stop_time = random.randint(action_stop, max_action_stop)

            for t in range(1000):

                if t < stop_time and t % delay == 0 :
                # get the best action according to q-network
                    with torch.no_grad():
                        state_t = torch.from_numpy(state).float().unsqueeze(0)
                        action_values = qnetwork(state_t)
                        action_values = softmax(action_values).cpu().data.numpy()[0]
                    action = np.argmax(action_values)
                elif t > stop_time:
                    action = 0

                # apply that action and take a step
                env.render()              # can always toggle visualization
                next_state, _, done, info = env.step(action)
                reward = info['reward']
                awake = info['awake']

                # Noise for noisy data - UT Austin method
                if r_type == 3 and np.random.random() < 0.05:
                    action = np.random.randint(0,4)
                xi.append([t] + [action] + [awake] + [state])

                state = next_state
                score += reward
                episode_reward += reward

                if done:
                    # print("Length of trajectory: ",len(xi))
                    print("Reward: ", episode_reward)
                    print("Lander Status: ",awake)
                    dataset.append(xi)
                    if awake:
                        landing += 1
                    break

        # scores[i] = score/episodes
        # landings[i] = landing
        # print("Mean reward for completed iteration: ", scores[i])

        env.close()
        dataname = '../data/lander_' + doc_name + '_t_' + str(delay) + '.pkl'
        # pickle.dump( dataset, open( dataname, "wb" ) )
        break

def main():
    '''
    Here i decides the type of data being generated
    i = 1 -> R1 data with no time stop (non counterfactual)
      = 2 -> R1 data with time stop (counterfactuals)
      = 3 -> R1 data with noise
      = 4 -> R2 data with no time stop
      = 5 -> R3 data with no time stop 
    '''
    # for i in range(1,5):
    #     if i == 2 or i == 3:
    #         eps = 100
    #     else:
    #         eps = 25
    eps = 25
    i = 4
    generate_data(i, eps)    

if __name__ == "__main__":
    main()

