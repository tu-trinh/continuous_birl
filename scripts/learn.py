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
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        else:
            shaping = \
                - 100*np.sqrt(state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
                - 100*abs(state[4]) + 10*state[6] + 10*state[7]

        if prev_shaping is not None:
            reward = shaping - prev_shaping
        prev_shaping = shaping
        
        if action == 1 or 3:
            reward -= 0.03
            
        elif action == 2:
            reward -= 0.30
            
        if not action == 0:
            reward += 1

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
    
    rewards_D = rewards_D/norm #np.max(np.abs(rewards_D))
    rewards_XiR = rewards_XiR/norm #np.max(np.abs(rewards_XiR))
    rewards_D_2 = rewards_D_2/norm #np.max(np.abs(rewards_D_2))
    rewards_XiR_2 = rewards_XiR_2/norm #np.max(np.abs(rewards_XiR_2))

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
    print("Belief: ", b)

def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "../data/lander_R1_t_5.pkl", "rb" ) )
    E = pickle.load( open( "../data/lander_R1_easy_t_5.pkl", "rb" ) )
    DR2 = pickle.load( open( "../data/lander_R2_t_1.pkl", "rb" ) )

    #build choice set --- default includes demonstrations and easy simplifications
    Xi_R = D + E

    Xi_R += DR2

    #rationality constant. Increasing makes different terms dominate
    for beta in [0, 0.1, 1, 5]:
        get_belief(beta, D, Xi_R)




if __name__ == "__main__":
    main()
