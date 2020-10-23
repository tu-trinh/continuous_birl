import gym
import cartpole_theta
import torch
import numpy as np
from train_optimal_agent import QNetwork
import pickle


def get_human(episodes, t_delay=8, type="regular"):
    env = gym.make("CartpoleTheta-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=0)
    qnetwork.load_state_dict(torch.load("models/cartpole_up.pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    dataset = []
    if type == "regular":
        stoptime_lb = 200
        noise_threshold = 0.0
    elif type == "noise":
        stoptime_lb = 200
        noise_threshold = 0.05
    elif type == "counterfactual":
        stoptime_lb = 0
        noise_threshold = 0.0
    for episode in range(episodes):
        stoptime = np.random.randint(stoptime_lb, 201)
        state = env.reset(theta="up")
        xi = []
        action = 0
        for t in range(500):
            if t < stoptime and t%t_delay == 0:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    action_values = qnetwork(state_t)
                    action_values = softmax(action_values).cpu().data.numpy()[0]
                action = np.argmax(action_values)
            if np.random.random() < noise_threshold:
                action = np.random.randint(0,2)
            xi.append([action] + list(state))
            # img = env.render(mode="rgb_array")              # can always toggle visualization
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                dataset.append(xi)
                break
    env.close()
    return dataset


def main():

    # play with the t_delay and number of demonstrations
    t_delay = 7
    demos = get_human(25, t_delay=t_delay, type="regular")

    N = 10
    noisies_set = []
    counterfactuals_set = []
    for i in range(0,N):
        noisies = get_human(100, t_delay=t_delay, type="noise")
        counterfactuals = get_human(100, t_delay=t_delay, type="counterfactual")
        noisies_set.append(noisies)
        counterfactuals_set.append(counterfactuals)

    pickle.dump( demos, open( "choices/demos.pkl", "wb" ) )
    pickle.dump( noisies_set, open( "choices/noisies_set.pkl", "wb" ) )
    pickle.dump( counterfactuals_set, open( "choices/counterfactuals_set.pkl", "wb" ) )


if __name__ == "__main__":
    main()
