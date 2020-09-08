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
    qnetwork.load_state_dict(torch.load("../models/dqn.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)

    # we'll rollout over N episodes
    episodes = 250

    ''' Calculate Rewards over multiple delays
        The action delay is calculated as timestep % delay_factor
    '''
    t_delay = range(1,17)
    scores = [None] * len(t_delay)
    landings = [None] * len(t_delay)

    # Trajectory Plotting
    f1, axs = plt.subplots(nrows=2, ncols=int(len(t_delay)/4), figsize=(20, 15))
    f1.suptitle('Sample trajectories w.r.t action delay')
    p_row = 0
    p_col = 0

    for i in range(len(t_delay)):
        score = 0
        landing = 0
        print("Starting lunar lander with action delay: ", t_delay[i])        
        for episode in range(episodes):
            if (episode%50 == 0):
                print("On episode: ", episode)
            # reset to start
            state = env.reset()
            state_x = []
            state_y = []
            episode_reward = 0
            for t in range(1000):

                if (t%t_delay[i] == 0) :
                # get the best action according to q-network
                    with torch.no_grad():
                        state_t = torch.from_numpy(state).float().unsqueeze(0)
                        action_values = qnetwork(state_t)
                        action_values = softmax(action_values).cpu().data.numpy()[0]
                    action = np.argmax(action_values)
                    #print(action)

                # apply that action and take a step
                #env.render()              # can always toggle visualization
                next_state, reward, done, info = env.step(action)
                state = next_state
                score += reward
                episode_reward += reward
                state_x.append(state[0])
                state_y.append(state[1]) 

                # Plot only half of the action delays
                if (episode < 15) and (i%2 == 0):
                    axs[p_row,p_col].plot(state_x, state_y)
                    axs[p_row,p_col].axis([-0.75, 0.75, 0, 1.6])
                    axs[p_row,p_col].text(0,1.55,'Action Delay: ' + str(t_delay[i]),
                                            horizontalalignment='center')
                    #axs[0].set(xlabel='Action Delay', ylabel='Mean Reward')
                    #axs[0].axhline(lw=1, color='black')
                if done:
                    if (episode_reward > 100):
                        landing += 1
                    break

        scores[i] = score/episodes
        landings[i] = landing
        print("Mean reward for completed iteration: ", scores[i])

        if(i%2 == 0):
            if (p_col < (len(t_delay)/4)-1):
                p_col += 1
            else:
                p_col = 0
                p_row += 1

        #print(score)
        #plt.plot(state_x,state_y)
        #plt.axis([-0.75, 0.75, 0, 1.6])
        

    # give some idea of how things went
    env.close()
    #print(scores)
    #print("mean score: ", np.mean(np.array(scores)))
    #print("std score: ", np.std(np.array(scores)))

    # Plots for rewards and landings
    f2, axs = plt.subplots(2)
    f2.suptitle('Mean reward and successful landings after ' + str(episodes) + ' episodes')
    axs[0].bar(t_delay, scores)
    axs[0].axis([0, max(t_delay), -250, 250])
    axs[0].set_title('Mean reward w.r.t action delay')
    axs[0].set(xlabel='Action Delay', ylabel='Mean Reward')
    axs[0].axhline(lw=1, color='black')


    axs[1].bar(t_delay, landings)
    axs[1].axis([0, max(t_delay), 0, episodes+1])
    axs[1].set_title('#Sucessful Landings w.r.t action delay')
    axs[1].set(xlabel='Action Delay', ylabel='Number of Successful Landings')


    f1.tight_layout()
    f2.tight_layout()
    f1.savefig('../plots/trajectories.png')
    f2.savefig('../plots/mean_reward.png')
    #plt.show()


if __name__ == "__main__":
    main()
