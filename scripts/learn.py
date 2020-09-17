# Script to simulate beliefs using methods from cartpole example

import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import pickle
import random



# Reward functions

def Reward(xi, R_type):
    R = 0
    shaping = 0
    prev_shaping = None
    reward = 0
    for i, waypoint in enumerate(xi):
        state = waypoint[3]
        action = waypoint[1]
        if R_type == 1:
            shaping = \
                - 200*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        else:
            shaping = \
                - 200*np.sqrt(state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]

        if prev_shaping is not None:
            reward = shaping - prev_shaping
        prev_shaping = shaping
        
        if action == 1 or 3:
            reward -= 0.03
            
        elif action == 2:
            reward -= 0.30
            
        # if not action == 0:
        #     reward += 1

        awake = waypoint[2]
        if i == len(waypoint) and awake:
            reward += 100
        elif not awake:
            reward -= 100
        R += reward
    return R

def get_belief(beta, D, Xi_R):

    # Reward for landing in the center
    rewards_D = np.asarray([Reward(xi,1) for xi in D], dtype = np.float32)    
    rewards_XiR = np.asarray([Reward(xi,1) for xi in Xi_R], dtype = np.float32)    
    rewards_D_2 = np.asarray([Reward(xi,2) for xi in D], dtype = np.float32)    
    rewards_XiR_2 = np.asarray([Reward(xi,2) for xi in Xi_R], dtype = np.float32)

    norm = np.max(np.asarray([np.max(np.abs(rewards_D)),np.max(np.abs(rewards_XiR)),\
         np.max(np.abs(rewards_D_2)), np.max(np.abs(rewards_XiR_2))]))
    
    rewards_D = rewards_D/np.max(np.abs(rewards_D))
    rewards_XiR = rewards_XiR/np.max(np.abs(rewards_XiR))
    rewards_D_2 = rewards_D_2/np.max(np.abs(rewards_D_2))
    rewards_XiR_2 = rewards_XiR_2/np.max(np.abs(rewards_XiR_2))

    # print("Rewards_D: ", rewards_D)
    # print("Rewards_XiR: ", rewards_XiR)
    # print("rewards_D_2: ", rewards_D_2)
    # print("Rewards_XiR_2: ", rewards_XiR_2)

    n1 = np.exp(beta*sum(rewards_D))
    d1 = sum(np.exp(beta*rewards_XiR))**len(D)
    p1 = n1/d1

    # Reward for landing anywhere
    n2 = np.exp(beta*sum(rewards_D_2))
    d2 = sum(np.exp(beta*rewards_XiR_2))**len(D)
    p2 = n2/d2

    Z = p1 + p2
    b = [p1/Z, p2/Z]
    return b

def birl_belief(beta, D, Xi_R1, Xi_R2):
    rewards_D = np.asarray([Reward(xi,1) for xi in D], dtype = np.float32)    
    rewards_XiR = np.asarray([Reward(xi,1) for xi in Xi_R1], dtype = np.float32)    
    rewards_D_2 = np.asarray([Reward(xi,2) for xi in D], dtype = np.float32)    
    rewards_XiR_2 = np.asarray([Reward(xi,2) for xi in Xi_R2], dtype = np.float32)

    norm = np.max(np.asarray([np.max(np.abs(rewards_D)),np.max(np.abs(rewards_XiR)),\
         np.max(np.abs(rewards_D_2)), np.max(np.abs(rewards_XiR_2))]))
    
    rewards_D = rewards_D/np.max(np.abs(rewards_D))
    rewards_XiR = rewards_XiR/np.max(np.abs(rewards_XiR))
    rewards_D_2 = rewards_D_2/np.max(np.abs(rewards_D_2))
    rewards_XiR_2 = rewards_XiR_2/np.max(np.abs(rewards_XiR_2))

    n1 = np.exp(beta*sum(rewards_D))
    d1 = np.exp(beta*sum(rewards_XiR))
    p1 = n1/d1

    # Reward for landing anywhere
    n2 = np.exp(beta*sum(rewards_D_2))
    d2 = np.exp(beta*sum(rewards_XiR_2))
    p2 = n2/d2

    Z = p1 + p2
    b = [p1/Z, p2/Z]
    return b


def main():

    b_e_t = []
    b_n_t = []
    b_o_t = []


    for t in range(1,11):
        f1, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 15))
        f1.suptitle('Belief wrt approach using constant delay: ' + str(t))
        width = 0.35
        row = 0
        col = 0


        #import trajectories (that could be choices)
        D = pickle.load( open( "../data/lander_R1_t_"+ str(t) +".pkl", "rb" ) )
        E = pickle.load( open( "../data/lander_R1_easy_t_"+ str(t) +".pkl", "rb" ) )
        O_R2 = pickle.load( open( "../data/lander_R2_t_1.pkl", "rb" ) )
        O_R1 = pickle.load( open("../data/lander_R1_t_1.pkl", "rb") )
        N = pickle.load( open( "../data/lander_R1_noisy_t_"+ str(t) +".pkl", "rb" ) )

        print("Action Delay: ", t)
        #rationality constant. Increasing makes different terms dominate
        b_e = []
        b_n = []
        b_o = []
        betas = [0.1, 0.5, 1, 5]
        Rewards = ['R1', 'R2']
        for beta in betas:
            # Our approach
            Xi_R = D + E
            b = get_belief(beta, D, Xi_R)
            b_e.append(b[0])
            


            # UT approach
            Xi_R = D + N
            b = get_belief(beta, D, Xi_R)
            b_n.append(b[0])
            # axs[col].bar(Rewards, b)
            # col += 1

            # Classic appraoch
            b = birl_belief(beta, D, O_R1, O_R2)
            b_o.append(b[0])
            # axs[col].bar(Rewards, b)
        titles = ['Our Approach', 'UT Approach', 'Classic Approach']
        for i, b in enumerate([b_e, b_n, b_o]):
            x_axis = np.arange(len(b))
            axs[col].bar(x_axis, b)
            axs[col].set_ylim([0, 1])
            axs[col].set_xticks(x_axis)
            axs[col].set_xticklabels(betas)
            axs[col].set_title(titles[i])
            axs[col].set_xlabel('Beta')
            axs[col].set_ylabel('Belief')
            col += 1   
        plt.savefig('../plots/modified_reward/beliefs_t_' + str(t) + '.png')
        # plt.show()

        b_e_t.append(b_e)
        b_n_t.append(b_n)
        b_o_t.append(b_o)
        
        
        


    


if __name__ == "__main__":
    main()
