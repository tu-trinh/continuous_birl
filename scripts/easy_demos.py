import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import LunarLander_c1
import pickle
import random

def get_params(r_type):
    model_name = "dqn.pth"
    # R1 with no time stop
    if r_type == 1:
        action_stop = 1000
        doc_name = 'R1'
    # R1 with time stops
    elif r_type == 2:
        action_stop = 1
        doc_name = 'R1_easy'
    # R2 with no time stops
    elif r_type == 3:
        action_stop = 1000
        doc_name = 'R1_noisy'

    else:
        action_stop = 1000
        doc_name = 'R2'
        model_name = "dqn_R2.pth"
    return action_stop, doc_name, model_name

def generate_data(r_type, ep, delay, max_action_stops):

    action_stop, doc_name, model_name = get_params(r_type)
    

    t_delay = range(1,11)
    scores = [None] * len(t_delay)
    modified_scores = [None] * len(t_delay)
    landings = [None] * len(t_delay)

    f1, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
    f1.suptitle('Sample easy trajectories w.r.t Max Action Stop')
    p_row = 0
    p_col = 0

    avg_dist = []
    sigma = []
    # Action delays. Decide how many frames between actions        

    for max_action_stop in max_action_stops:
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

        dist_episode = []

        for episode in range(episodes):   
            
            state = env.reset()

            # Some useful variables
            landing = 0
            xi = []
            episode_reward = 0
            modified_episode_reward = 0
            
            # Save x,y for plotting
            state_x = []
            state_y = []

            # Stop time for easy demonstrations
            stop_time = random.randint(action_stop, max_action_stop)
            #print(stop_time)

            # Run demonstrations for all t
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
                # env.render()              # can always toggle visualization
                next_state, _, done, info = env.step(action)
                reward = info['reward']
                modified_reward = info['mod_reward']
                awake = info['awake']

                # Noise for noisy data - UT Austin method
                if r_type == 3 and np.random.random() < 0.05:
                    action = np.random.randint(0,4)
                xi.append([t] + [action] + [awake] + [state])

                state = next_state
                score += reward
                modified_score += modified_reward
                episode_reward += reward
                modified_episode_reward += modified_reward
                state_x.append(state[0])
                state_y.append(state[1])

                if done:
                    # print("Length of trajectory: ",len(xi))
                    # print("Reward: ", episode_reward)
                    # print("Lander Status: ",awake)
                    if episode < 15:
                        axs[p_row,p_col].plot(state_x, state_y)
                        axs[p_row,p_col].axis([-0.75, 0.75, 0, 1.6])
                        axs[p_row,p_col].text(0,1.55,'Max Stop Time: ' + str(max_action_stop),
                                                 horizontalalignment='center')

                    dataset.append(xi)
                    dist_episode.append(state[0])
                    if awake:
                        landing += 1
                    break

        # scores[i] = score/episodes
        # modified_scores[i] = modified_score/episodes
        # landings[i] = landing
        # print("Mean reward for completed iteration: ", scores[i])

        env.close()
        if p_col < 2:
            p_col += 1
        else:
            p_col = 0
            p_row += 1
        avg_dist.append(sum(dist_episode)/len(dist_episode))
        sigma.append(np.var(dist_episode))
        # dataname = '../data/lander_' + doc_name + '_t_' + str(delay) + '.pkl'
        # pickle.dump( dataset, open( dataname, "wb" ) )
        #print(dataset)
        #print(dataset[0])
        #print(len(dataset))
    f1.tight_layout()
    f1.savefig('../plots/easy_traj/trajectories_t_' + str(delay) + '.png')
    return avg_dist, sigma
    # plt.show()

def main():
    '''
    Here i decides the type of data being generated
    i = 1 -> R1 data with no time stop (non counterfactual)
      = 2 -> R1 data with time stop (counterfactuals)
      = 3 -> R1 data with noise
      = 4 -> R2 data with no time stop 
    '''
    avg_dist = []
    t_delay = range(1,11)
    max_action_stops = [100, 200, 300, 400, 500, 600]
    eps = 25
    r_type = 2
    x = np.arange(len(max_action_stops))

    f2, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 15))
    f2.suptitle('Avg landing distance w.r.t Max action stop')
    p_row = 0
    p_col = 0
    width = 0.35

    for delay in t_delay:
        # i = 1
        dist, sigma = generate_data(r_type, eps, delay, max_action_stops)   
        rects1 = axs[p_row,p_col].bar(x - width/2, dist, 0.35, label = 'Mean')
        rects2 = axs[p_row,p_col].bar(x + width/2, sigma, 0.35, label = 'Variance')
        # print(dist)
        axs[p_row,p_col].title.set_text('Action Delay: ' + str(delay))
        axs[p_row,p_col].set_xticks(x)
        axs[p_row,p_col].set_xticklabels(max_action_stops)
        axs[p_row,p_col].set(xlabel='Max Action Stop', ylabel='Avg Distance')
        axs[p_row,p_col].axhline(lw=1, color='black')
        axs[p_row,p_col].legend()
        axs[p_row,p_col].set_ylim([-0.2, 0.2])

        if p_col < 4:
            p_col += 1
        else:
            p_col = 0
            p_row += 1        

    f2.tight_layout()
    f2.savefig('../plots/avg_dist.png')
if __name__ == "__main__":
    main()

