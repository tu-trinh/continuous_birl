import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import pickle
import random

# Reward functions
def Reward(xi, r_type):
    R = 0
    shaping = 0
    prev_shaping = None
    reward = 0
    initial_waypoint = xi[0]
    initial_state = initial_waypoint[3]
    initial_x = initial_state[0]
    for i, waypoint in enumerate(xi):
        state = waypoint[3]
        action = waypoint[1]
        if r_type == 1:
            shaping = \
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        elif r_type == 2:
            shaping = \
                - 100*np.sqrt(state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        else:
            shaping = \
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 0.0*abs(state[4]) + 0.0*state[6] + 0.0*state[7]

        if prev_shaping is not None:
            reward = shaping - prev_shaping
        prev_shaping = shaping

        if action == 1 or 3:
            reward -= 0.03

        elif action == 2:
            reward -= 0.30

        awake = waypoint[2]
        if i == len(waypoint) and awake:
            if np.linalg.norm(state[0]) < 0.1 and r_type == 1:
                reward = +100
            if np.linalg.norm(state[0] - initial_x) < 0.1 and r_type == 2:
                reward = +100
            else:
                reward = -100
        elif not awake:
            if r_type == 3:
                reward = +100
            else:
                reward = -100
        R += reward
    return R

def get_belief(beta, D, Xi_R):
    rewards_D_1 = np.asarray([Reward(xi,1) for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,1) for xi in Xi_R], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,2) for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,2) for xi in Xi_R], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,3) for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,3) for xi in Xi_R], dtype = np.float32)

    # Reward for landing in middle
    n1 = np.exp(beta*sum(rewards_D_1))
    d1 = sum(np.exp(beta*rewards_XiR_1))**len(D)
    p1 = n1/d1

    # Reward for landing anywhere
    n2 = np.exp(beta*sum(rewards_D_2))
    d2 = sum(np.exp(beta*rewards_XiR_2))**len(D)
    p2 = n2/d2

    # Reward for crashing in middle
    n3 = np.exp(beta*sum(rewards_D_3))
    d3 = sum(np.exp(beta*rewards_XiR_3))**len(D)
    p3 = n3/d3

    Z = p1 + p2 + p3
    b = [p1/Z, p2/Z, p3/Z]
    return b

def birl_belief(beta, D, Xi_R1, Xi_R2, Xi_R3):
    rewards_D_1 = np.asarray([Reward(xi,1) for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,1) for xi in Xi_R1], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,2) for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,2) for xi in Xi_R2], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,3) for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,3) for xi in Xi_R3], dtype = np.float32)

    # Reward for landing in middle
    n1 = np.exp(beta*sum(rewards_D_1))
    d1 = np.exp(beta*sum(rewards_XiR_1))
    p1 = n1/d1

    # Reward for landing anywhere
    n2 = np.exp(beta*sum(rewards_D_2))
    d2 = np.exp(beta*sum(rewards_XiR_2))
    p2 = n2/d2

    # Reward for crashing in the middle
    n3 = np.exp(beta*sum(rewards_D_3))
    d3 = np.exp(beta*sum(rewards_XiR_3))
    p3 = n3/d3

    Z = p1 + p2 + p3
    b = [p1/Z, p2/Z, p3/Z]
    return b


def main():

    B_ours = np.zeros((10,3))
    B_ut = np.zeros((10,3))
    B_classic = np.zeros((10,3))

    for t in range(1,11):
        #import trajectories (that could be choices)
        D = pickle.load( open( "../data/lander_R1_t_"+ str(t) +".pkl", "rb" ) )
        E = pickle.load( open( "../data/lander_R1_easy_t_"+ str(t) +".pkl", "rb" ) )
        O_R1 = pickle.load( open("../data/lander_R1_t_1.pkl", "rb") )
        O_R2 = pickle.load( open("../data/lander_R2_t_1.pkl", "rb" ) )
        O_R3 = pickle.load( open("../data/lander_R3_t_1.pkl", "rb") )
        N = pickle.load( open( "../data/lander_R1_noisy_t_"+ str(t) +".pkl", "rb" ) )

        beta = 0.001

        #Classic approach
        b_cl = birl_belief(beta, D, O_R1, O_R2, O_R3)
        B_classic[t-1,:] = np.asarray(b_cl)

        # Our approach
        Xi_R = D + E
        b_our = get_belief(beta, D, Xi_R)
        B_ours[t-1,:] = np.asarray(b_our)

        # UT approach
        Xi_R = D + N
        b_ut = get_belief(beta, D, Xi_R)
        B_ut[t-1,:] = np.asarray(b_ut)

    plt.plot(B_classic,'x-')
    plt.title("belief with classic approach")
    plt.xlabel("time delay")
    plt.ylabel("belief")
    plt.legend(["R1", "R2", "R3"])
    plt.show()

    plt.plot(B_ours,'o-')
    plt.title("belief with our approach")
    plt.xlabel("time delay")
    plt.ylabel("belief")
    plt.legend(["R1", "R2", "R3"])
    plt.show()

    plt.plot(B_ut,'s-')
    plt.title("belief with noise approach")
    plt.xlabel("time delay")
    plt.ylabel("belief")
    plt.legend(["R1", "R2", "R3"])
    plt.show()




    # for t in range(1,11):
    #     f1, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    #     f1.suptitle('Belief wrt approach using constant delay: ' + str(t))
    #     width = 0.35
    #     row = 0
    #     col = 0
    #
    #     #import trajectories (that could be choices)
    #     D = pickle.load( open( "../data/lander_R1_t_"+ str(t) +".pkl", "rb" ) )
    #     E = pickle.load( open( "../data/lander_R1_easy_t_"+ str(t) +".pkl", "rb" ) )
    #     O_R1 = pickle.load( open("../data/lander_R1_t_1.pkl", "rb") )
    #     O_R2 = pickle.load( open("../data/lander_R2_t_1.pkl", "rb" ) )
    #     O_R3 = pickle.load( open("../data/lander_R3_t_1.pkl", "rb") )
    #     N = pickle.load( open( "../data/lander_R1_noisy_t_"+ str(t) +".pkl", "rb" ) )
    #
    #     print("Action Delay: ", t)
    #     #rationality constant. Increasing makes different terms dominate
    #     betas = [0.1, 0.5, 0.75, 1]
    #     Rewards = ['R1', 'R2']
    #
    #     b_our_R1 = []
    #     b_our_R2 = []
    #     b_our_R3 = []
    #     b_ut_R1 = []
    #     b_ut_R2 = []
    #     b_ut_R3 = []
    #     b_cl_R1 = []
    #     b_cl_R2 = []
    #     b_cl_R3 = []
    #     for beta in betas:
    #         # Our approach
    #         Xi_R = D + E
    #         b_our = get_belief(beta, D, Xi_R)
    #         b_our_R1.append(b_our[0])
    #         b_our_R2.append(b_our[1])
    #         b_our_R3.append(b_our[2])
    #
    #         # UT approach
    #         Xi_R = D + N
    #         b_ut = get_belief(beta, D, Xi_R)
    #         b_ut_R1.append(b_ut[0])
    #         b_ut_R2.append(b_ut[1])
    #         b_ut_R3.append(b_ut[2])
    #
    #         #Classic approach
    #         b_cl = birl_belief(beta, D, O_R1, O_R2, O_R3)
    #         b_cl_R1.append(b_cl[0])
    #         b_cl_R2.append(b_cl[1])
    #         b_cl_R3.append(b_cl[2])
    #
    #     titles = ['Our Approach', 'UT Approach', 'Classic Approach']
    #     for i, b in enumerate([b_our_R1, b_ut_R1, b_cl_R1]):
    #         x_axis = np.arange(len(betas))
    #         axs[0,col].bar(x_axis, b)
    #         axs[0,col].set_ylim([0, 1])
    #         axs[0,col].set_xticks(x_axis)
    #         axs[0,col].set_xticklabels(betas)
    #         axs[0,col].set_title(titles[i])
    #         axs[0,col].set_xlabel('Beta')
    #         axs[0,col].set_ylabel('Belief')
    #         col += 1
    #     col = 0
    #     for i, b in enumerate([b_our_R2, b_ut_R2, b_cl_R2]):
    #         x_axis = np.arange(len(betas))
    #         axs[1,col].bar(x_axis, b)
    #         axs[1,col].set_ylim([0, 1])
    #         axs[1,col].set_xticks(x_axis)
    #         axs[1,col].set_xticklabels(betas)
    #         axs[1,col].set_title(titles[i])
    #         axs[1,col].set_xlabel('Beta')
    #         axs[1,col].set_ylabel('Belief')
    #         col += 1
    #     col = 0
    #     for i, b in enumerate([b_our_R3, b_ut_R3, b_cl_R3]):
    #         x_axis = np.arange(len(betas))
    #         axs[2,col].bar(x_axis, b)
    #         axs[2,col].set_ylim([0, 1])
    #         axs[2,col].set_xticks(x_axis)
    #         axs[2,col].set_xticklabels(betas)
    #         axs[2,col].set_title(titles[i])
    #         axs[2,col].set_xlabel('Beta')
    #         axs[2,col].set_ylabel('Belief')
    #         col += 1
    #     f1.tight_layout()
    #     f1.savefig('../plots/modified_reward/beliefs_t_' + str(t) + '.png')
    #     plt.show()

if __name__ == "__main__":
    main()
