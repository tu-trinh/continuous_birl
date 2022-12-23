import numpy as np
import pickle
from train_optimal_agent import Agent, train
from collect_optimal_demos import gen_traj
from env.lunarlander_theta.envs.lunarlander_theta import LunarLanderTheta


### Code to generate hypotheses below ###
def generate_hypotheses():
    hypotheses = {
        "center": [0, 0, -100, 0, 0, -100, 0, -100, 0, 10, 10],
        "anywhere": [0, -100, 0, 0, 0, -100, 0, -100, 0, 10, 10],
        "crash": [0, 0, -100, 0, 0, -100, 0, 0, 0, 0, 0]
    }
    for i in range(97): # assuming we want 100 thetas, 3 of which are center, anywhere, and crash
        type = np.random.choice(["center", "anywhere", "crash"])
        if type == "center":
            hypotheses["hypo{}".format(i)] = [0, 0, np.random.randint(-150, -50), 0, 0, np.random.randint(-150, -50), 0, np.random.randint(-150, -50), 0, np.random.randint(1, 20), np.random.randint(1, 20)]
        elif type == "anywhere":
            hypotheses["hypo{}".format(i)] = [0, np.random.randint(-150, -50), 0, 0, 0, np.random.randint(-150, -50), 0, np.random.randint(-150, -50), 0, np.random.randint(1, 20), np.random.randint(1, 20)]
        else:
            hypotheses["hypo{}".format(i)] = [0, 0, np.random.randint(-150, -50), 0, 0, np.random.randint(-150, -50), 0, 0, 0, 0, 0]
    print(hypotheses)


### Code to train DQNs for each hypothesis below ###
def train_DQNs():
    f = open("hypotheses.txt", "r")
    hypotheses = eval(f.read())
    for theta in hypotheses:
        if theta != "center" and theta != "anywhere" and theta != "crash":
            env = LunarLanderTheta()
            env.seed(0)
            agent = Agent(state_size=8, action_size=4, seed=0)
            train(agent, env, theta = theta, n_episodes = 600, max_t = 100)
            print("Done training", theta)


### Code to generate policies for hypotheses below ###
def generate_policies():
    f = open("hypotheses.txt", "r")
    hypotheses = eval(f.read())
    hypo_policies = {}
    for theta in hypotheses:
        if theta != "center" and theta != "anywhere" and theta != "crash":
            hypo_policies[theta] = gen_traj(15, theta = theta)
            print("Done {}".format(theta))
    pickle.dump(hypo_policies, open("hypo_policies.pkl", "wb"))


if __name__ == "__main__":
    # generate_hypotheses()
    # train_DQNs()
    generate_policies()