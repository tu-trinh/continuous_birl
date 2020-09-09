import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt


def main():
    
    fig, ax = plt.subplots()
    width = 0.35
    t_delay = range(1,11)
    scores = [None] * len(t_delay)
    modified_scores = [None] * len(t_delay)
    landings = [None] * len(t_delay)
    

    t_delay_2 = [delay + width for delay in t_delay]
    for iteration in range(2):
        # load environment
        env = gym.make('LunarLander-v2')

        # load our trained q-network
        qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
        qnetwork.load_state_dict(torch.load("../models/dqn.pth"))
        qnetwork.eval()
        softmax = torch.nn.Softmax(dim=1)

        # we'll rollout over N episodes
        episodes = 500

        # R1 and R2 with R1 reward plotting


        for i in range(len(t_delay)):
            score = 0
            modified_score = 0
            landing = 0
            print("Starting lunar lander with action delay: ", t_delay[i])
            for episode in range(episodes):
                if (episode%50 == 0):
                    print("On episode: ", episode)
                # reset to start
                state = env.reset()
                
                if(iteration == 1):
                    env.set_custom_reward(True)
                    
                state_x = []
                state_y = []
                episode_reward = 0
                modified_episode_reward = 0
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
                    modified_reward = env.get_modified_reward()
                    state = next_state
                    score += reward
                    modified_score += modified_reward
                    episode_reward += reward
                    modified_episode_reward += modified_reward
                    state_x.append(state[0])
                    state_y.append(state[1])

                    if done:
                        if (episode_reward > 100):
                            landing += 1
                        break

            scores[i] = score/episodes
            modified_scores[i] = modified_score/episodes
            landings[i] = landing
            print("Mean reward for completed iteration: ", scores[i])

            # give some idea of how things went
            env.close()

        if(iteration == 0):
            p1 = ax.bar(t_delay, scores, width, bottom=0)
        else:
            p2 = ax.bar(t_delay_2, scores, width, bottom=0)

    ax.set_title('Mean reward for R1 network wrt to R1 and R2 rewards after ' 
        + str(episodes) + ' episodes')
    #t_delay_2 = [delay/2 for delay in t_delay_2]
    ax.set_xticks(t_delay_2)
    ax.set_xticklabels(t_delay)
    ax.legend((p1[0], p2[0]), ('R1', 'R2'))
    ax.set(xlabel='Action Delay', ylabel='Reward')
    ax.axhline(lw=1, color='black')
    plt.show()
        #print(scores)
        #print("mean score: ", np.mean(np.array(scores)))
        #print("std score: ", np.std(np.array(scores)))

    # # Plots for rewards and landings
    # f2, axs = plt.subplots(2)
    # f2.suptitle('Mean reward and successful landings after ' + str(episodes) + ' episodes')
    # axs[0].bar(t_delay, scores)
    # axs[0].axis([0, max(t_delay), -250, 250])
    # axs[0].set_title('Mean reward w.r.t action delay')
    # axs[0].set(xlabel='Action Delay', ylabel='Mean Reward')
    # axs[0].axhline(lw=1, color='black')


    # axs[1].bar(t_delay, landings)
    # axs[1].axis([0, max(t_delay), 0, episodes+1])
    # axs[1].set_title('#Sucessful Landings w.r.t action delay')
    # axs[1].set(xlabel='Action Delay', ylabel='Number of Successful Landings')


    # f1.tight_layout()
    # f2.tight_layout()
    # f1.savefig('../plots/trajectories.png')
    # f2.savefig('../plots/mean_reward.png')
    # #plt.show()


if __name__ == "__main__":
    main()
