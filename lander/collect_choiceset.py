import gym
import cartpole_theta
import torch
import numpy as np
from train_optimal_agent import QNetwork
import pickle


def gen_traj(episodes, t_delay=8, theta=None):
    env = gym.make('LunarLanderTheta-v0')
    path = "models/dqn_" + theta + ".pth"
    qnetwork = QNetwork(state_size=8, action_size=4, seed=1)
    qnetwork.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    noise_threshold = 0.2

    for episode in range(episodes):
        state = env.reset(theta=theta)
        xi = []
        action = 0
        episode_reward = 0

        for t in range(1000):
            if t%t_delay == 0:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    action_values = qnetwork(state_t)
                    action_values = softmax(action_values).cpu().data.numpy()[0]
                action = np.argmax(action_values)
            if np.random.random() < noise_threshold and t < 200:
                action = np.random.randint(0,4)
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

    N = 100
    t_delay = 5

    choiceset = []
    # choiceset += gen_traj(N, t_delay, theta="center")
    choiceset += gen_traj(N, t_delay, theta="anywhere")
    choiceset += gen_traj(N, t_delay, theta="crash")

    pickle.dump( choiceset, open( "choices/choiceset.pkl", "wb" ) )


if __name__ == "__main__":
    main()
