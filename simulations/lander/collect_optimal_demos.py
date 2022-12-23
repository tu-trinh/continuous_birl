import gym
from env.lunarlander_theta.envs.lunarlander_theta import LunarLanderTheta
import torch
import numpy as np
from train_optimal_agent import QNetwork
import pickle


def gen_traj(episodes, theta=None):
    # load environment
    # env = gym.make('LunarLanderTheta-v0')
    env = LunarLanderTheta()
    # load our trained q-network
    path = "models/dqn_" + theta + ".pth"
    qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
    qnetwork.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []

    for episode in range(episodes):
        state = env.reset(theta=theta)
        xi = []
        episode_reward = 0

        for t in range(1000):
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)
            # env.render()              # can always toggle visualization
            next_state, _, done, info = env.step(action)
            awake = info["awake"]
            reward = info["reward"]
            xi.append([t] + [action] + [awake] + [state])
            state = next_state
            episode_reward += reward

            if done:
                print("\rReward: {:.2f}\tLanded: {}\tReward: {}"\
                .format(episode_reward, awake, theta), end="")
                dataset.append(xi)
                break
    env.close()
    return dataset


def main():
    episodes = 25
    optimals = {'center': [], 'anywhere': [], 'crash': []}
    optimals['center'] = gen_traj(episodes, theta="center")
    optimals['anywhere'] = gen_traj(episodes, theta="anywhere")
    optimals['crash'] = gen_traj(episodes, theta="crash")
    pickle.dump( optimals, open( "choices/optimal.pkl", "wb" ) )


if __name__ == "__main__":
    main()
