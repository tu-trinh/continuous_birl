import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import LunarLander_c1
import pickle
import random



def plot_traj(states, max_action_stops, delay):
    nrows = 2
    ncols = 3
    f1, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
    f1.suptitle('Sample easy trajectories w.r.t Max Action Stop')
    p_row = 0
    p_col = 0

    for i, states_episode in enumerate(states):
        for state in states_episode:
            axs[p_row,p_col].plot(state[0], state[1])
            axs[p_row,p_col].axis([-0.75, 0.75, 0, 1.6])
            if max_action_stops[i] is not 1000:
                axs[p_row,p_col].text(0,1.55,'Max Stop Time: ' + str(max_action_stops[i]),
                                 horizontalalignment='center')
            else:
                axs[p_row,p_col].text(0,1.55,'Max Stop Time: No Stop Time',
                                 horizontalalignment='center')
        if p_col < ncols - 1:
            p_col += 1
        else:
            p_col = 0
            p_row += 1
    f1.tight_layout()
    f1.savefig('../plots/easy_traj/trajectories_t_' + str(delay) + '.png')


def generate_traj(plot_flag, ep, delay, max_action_stops):

    action_stop = 1 
    model_name = 'dqn_R1.pth'
    t_delay = range(1,11)
    scores = [None] * len(t_delay)
    landings = [None] * len(t_delay)
    avg_dist = []
    sigma = []
    states = []

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
        print("\rStarting lunar lander with action delay: ", delay)

        dist_episode = []
        states_episode = []

        for episode in range(episodes):   
            
            state = env.reset(reward_type=1)

            # Some useful variables
            landing = 0
            episode_reward = 0
            state_x = []
            state_y = []

            # Stop time for easy demonstrations
            if max_action_stop == 1000:
                action_stop = max_action_stop

            stop_time = random.randint(action_stop, max_action_stop)

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
                awake = info['awake']
                state = next_state
                score += reward
                episode_reward += reward
                state_x.append(state[0])
                state_y.append(state[1])

                if done:
                    if episode < 15:
                        states_episode.append([state_x, state_y])
                    dist_episode.append(state[0])
                    if awake:
                        landing += 1
                    break
        env.close()
        states.append(states_episode)
        avg_dist.append(sum(dist_episode)/len(dist_episode))
        sigma.append(np.var(dist_episode))

    if plot_flag:
        plot_traj(states, max_action_stops, delay)

    return avg_dist, sigma
    # plt.show()

def plot_dist(dists, sigmas, max_action_stops, t_delay):
        x = np.arange(len(max_action_stops))
        f2, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 15))
        f2.suptitle\
        ('Avg landing distance w.r.t Max action stop (Max Action Stop 1000 = No stop time)')
        p_row = 0
        p_col = 0
        width = 0.35
        for i in range(len(dists)):
            axs[p_row,p_col].bar(x-width/2, dists[i], 0.35, label = 'Mean')
            axs[p_row,p_col].bar(x+width/2, sigmas[i], 0.35, label = 'variance')
            axs[p_row,p_col].title.set_text('Action Delay: ' + str(t_delay[i]))
            axs[p_row,p_col].set_xticks(x)
            axs[p_row,p_col].set_xticklabels(max_action_stops)
            axs[p_row,p_col].set(xlabel='Max Action Stop', ylabel='Avg Distance')
            axs[p_row,p_col].axhline(lw=1, color='black')
            axs[p_row,p_col].legend()
            axs[p_row,p_col].set_ylim([-0.25, 0.25])

            if p_col < 4:
                p_col += 1
            else:
                p_col = 0
                p_row += 1        
        f2.tight_layout()
        f2.savefig('../plots/avg_dist.png')

def main():
    dists = []
    sigmas = []
    t_delay = range(1,11)
    max_action_stops = [100, 200, 300, 400, 500, 1000]
    eps = 25
    plot_flag = True

    for delay in t_delay:
        dist, sigma = generate_traj(plot_flag, eps, delay, max_action_stops)
        dists.append(dist)
        sigmas.append(sigma)
    
    if plot_flag:
        plot_dist(dists, sigmas, max_action_stops, t_delay)

if __name__ == "__main__":
    main()

