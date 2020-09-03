import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import pickle
import random


#reward function for keeping pole vertical
def Rup(xi):
    R = 0
    for waypoint in xi:
        angle = waypoint[4]
        if abs(abs(angle) - 0.0) < 0.1:
            R += 1
    return R / 200

#reward function for moving right (action = 1)
def Rright(xi):
    R = 0
    for waypoint in xi:
        action = waypoint[1]
        R = R + action
    return R / 200

#reward function for moving left (action = 0)
def Rleft(xi):
    R = 0
    for waypoint in xi:
        action = waypoint[1]
        R = R + 1 - action
    return R / 200

#reward function for keeping pole at an angle of pi/12
def Rtilt(xi):
    R = 0
    for waypoint in xi:
        angle = waypoint[4]
        if abs(abs(angle) - np.pi/12) < 0.1:
            R += 1
    return R / 200



def main():

    #import trajectories (that could be choices)
    D = pickle.load( open( "demo.pkl", "rb" ) )
    E = pickle.load( open( "easy.pkl", "rb" ) )
    DstarUP = pickle.load( open( "demo0.pkl", "rb" ) )
    DstarTILT = pickle.load( open( "demo1.pkl", "rb" ) )
    DstarLEFT = pickle.load( open( "demo2.pkl", "rb" ) )
    DstarRIGHT = pickle.load( open( "demo3.pkl", "rb" ) )

    #build choice set --- default includes demonstrations and easy simplifications
    Xi_R = D + E

    #optionally, you can also include optimal trajectories for each reward function.
    #if the human had no limitations, we would expect them to show one of these!
    # Xi_R += DstarUP
    # Xi_R += DstarRIGHT
    # Xi_R += DstarLEFT
    # Xi_R += DstarTILT

    #rationality constant. Increasing makes different terms dominate
    beta = 20

    #bayesian inference for each reward given demonstrations and choice set

    #reward for keeping pole vertical
    n1 = np.exp(beta*sum([Rup(xi) for xi in D]))
    d1 = sum([np.exp(beta*Rup(xi)) for xi in Xi_R])**len(D)
    p1 = n1/d1

    #reward for moving cart right
    n2 = np.exp(beta*sum([Rright(xi) for xi in D]))
    d2 = sum([np.exp(beta*Rright(xi)) for xi in Xi_R])**len(D)
    p2 = n2/d2

    #reward for moving cart left
    n3 = np.exp(beta*sum([Rleft(xi) for xi in D]))
    d3 = sum([np.exp(beta*Rleft(xi)) for xi in Xi_R])**len(D)
    p3 = n3/d3

    #reward for keeping pole at an angle
    n4 = np.exp(beta*sum([Rtilt(xi) for xi in D]))
    d4 = sum([np.exp(beta*Rtilt(xi)) for xi in Xi_R])**len(D)
    p4 = n4/d4

    #normalize and print belief
    Z = p1 + p2 + p3 + p4
    b = [p1/Z, p2/Z, p3/Z, p4/Z]
    print("Belief: ", b)


if __name__ == "__main__":
    main()
