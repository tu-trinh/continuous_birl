import gym
import torch
import numpy as np
from train import QNetwork
import matplotlib.pyplot as plt
import LunarLander_c1
import pickle
import random

def get_params(r_type, d_type, max_action_stop):
    model_name = "dqn_R" + str(r_type) + ".pth"
    action_stop = max_action_stop
    # No time stop
    if d_type == 1:
        doc_name = 'R' + str(r_type)
    # With time stops
    elif d_type == 2:
        action_stop = 1
        doc_name = 'R' + str(r_type) + '_easy'
    # With noise
    elif d_type == 3:
        doc_name = 'R' + str(r_type) +'_noisy'
    return action_stop, doc_name, model_name


def gen_traj(ep, delay=1, d_type=1, r_type=1, max_action_stop = 1000, save_data=False):

    action_stop, doc_name, model_name = get_params(r_type, d_type, max_action_stop)

    # load our trained q-network
    path = "../models/" + model_name
    qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
    qnetwork.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)

    scores = None
    landings = None
    dataset = []
    # load environment
    env = gym.make('LunarLanderC1-v0')

    # we'll rollout over N episodes
    episodes = ep

    score = 0
    for episode in range(episodes):
        # reset to start
        state = env.reset(reward_type = r_type)
        landing = 0
        xi = []
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
            # Noise for noisy data - UT Austin method
            if d_type == 3 and np.random.random() < 0.4 and t < 200:
                action = np.random.randint(0,4)

            # apply that action and take a step
            # env.render()              # can always toggle visualization
            next_state, _, done, info = env.step(action)
            reward = info['reward']
            awake = info['awake']
            rewards = info['rewards']
            xi.append([t] + [action] + [awake] + [state])
            state = next_state
            score += reward
            episode_reward += reward

            if done:
                print("\rReward: {:.2f}\tLanded: {}\tData Type: {}"\
                .format(episode_reward, awake, r_type, doc_name), end="")
                # print("Stoptime: {}\tCurrent Time: {}".format(stop_time, t))
                dataset.append(xi)
                if awake:
                    landing += 1
                break
    env.close()
    if save_data:
        savename = doc_name + '_t_' + str(delay)
        print("\nSave name {}\tEpisodes: {}".format(savename, len(dataset)))
        dataname = '../data/lander_' + savename + '.pkl'
        pickle.dump( dataset, open( dataname, "wb" ) )
    return dataset

def main():

    t_delay = range(1,11)
    episodes = 25
    episodes_noisy = 100
    episodes_counter = 100
    max_action_stop = 150
    save_data = True

    for delay in t_delay:
        r1_demos = gen_traj(episodes, delay=delay, d_type=1, save_data=save_data)
        r1_counterfactuals = gen_traj(episodes_counter, delay=delay,\
                             d_type=2, max_action_stop=max_action_stop, save_data=save_data)
        r1_noisies = gen_traj(episodes_noisy, delay=delay, d_type=3, save_data=save_data)

    r2_optimals = gen_traj(episodes, r_type=2, save_data=save_data)
    r3_optimals = gen_traj(episodes, r_type=3, save_data=save_data)

if __name__ == "__main__":
    main()
