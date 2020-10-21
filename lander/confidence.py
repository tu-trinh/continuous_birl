import gym
import lunarlander_theta
import torch
import numpy as np
from train_optimal_agent import QNetwork
import pickle


def gen_traj(episodes, t_delay=8, theta=None):

    # load environment
    env = gym.make('LunarLanderTheta-v0')

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
            if t%t_delay == 0:
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

def birl_belief(beta, D, O):
    rewards_D_1 = np.asarray([Reward(xi,"center") for xi in D], dtype = np.float32)
    rewards_XiR_1 = np.asarray([Reward(xi,"center") for xi in O["center"]], dtype = np.float32)
    rewards_D_2 = np.asarray([Reward(xi,"anywhere") for xi in D], dtype = np.float32)
    rewards_XiR_2 = np.asarray([Reward(xi,"anywhere") for xi in O["anywhere"]], dtype = np.float32)
    rewards_D_3 = np.asarray([Reward(xi,"crash") for xi in D], dtype = np.float32)
    rewards_XiR_3 = np.asarray([Reward(xi,"crash") for xi in O["crash"]], dtype = np.float32)

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

    episodes = 25
    t_delay = 10
    confidence = {'center': [], 'anywhere': [], 'crash': []}
    confidence['center'] = gen_traj(episodes, theta="center")
    confidence['anywhere'] = gen_traj(episodes, theta="anywhere")
    confidence['crash'] = gen_traj(episodes, theta="crash")
    pickle.dump( confidence, open( "choices/confidence.pkl", "wb" ) )

    demos = pickle.load( open( "choices/demos.pkl", "rb") )




if __name__ == "__main__":
    main()
