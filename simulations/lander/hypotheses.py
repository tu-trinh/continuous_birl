import numpy as np
import pickle
from train_optimal_agent import Agent, train
from collect_optimal_demos import gen_traj
from env.lunarlander_theta.envs.lunarlander_theta import LunarLanderTheta

### Code to generate hypotheses below ###
def generate_hypotheses():
    hypotheses = {
        # "center": [0, 0, -100, 0, 0, -100, 0, -100, 0, 10, 10],
        # "anywhere": [0, -100, 0, 0, 0, -100, 0, -100, 0, 10, 10],
        # "crash": [0, 0, -100, 0, 0, -100, 0, 0, 0, 0, 0]
    }
    for i in range(140, 200):
        type = np.random.choice(["center", "anywhere", "crash"])
        if type == "center":
            hypotheses["hypo{}".format(i)] = [0, 0, np.random.randint(-150, -50), 0, 0, np.random.randint(-150, -50), 0, np.random.randint(-150, -50), 0, np.random.randint(1, 20), np.random.randint(1, 20)]
        elif type == "anywhere":
            hypotheses["hypo{}".format(i)] = [0, 0, np.random.randint(-250, -150), 0, 0, np.random.randint(-100, 0), 0, np.random.randint(-250, -150), 0, np.random.randint(1, 10), np.random.randint(1, 10)]
        else:
            hypotheses["hypo{}".format(i)] = [0, 0, np.random.randint(-350, -250), 0, 0, np.random.randint(-60, 40), 0, np.random.randint(-350, -250), 0, np.random.randint(1, 15), np.random.randint(1, 15)]
    print(hypotheses)


### Code to train DQNs for each hypothesis below ###
def train_DQNs():
    f = open("working_hypotheses.txt", "r")
    hypotheses = eval(f.read())
    for theta in hypotheses:
        env = LunarLanderTheta()
        env.seed(0)
        agent = Agent(state_size=8, action_size=4, seed=0)
        train(agent, env, theta = theta, n_episodes = 600, max_t = 100)
        print("Done training", theta)


### Code to generate policies for hypotheses below ###
def generate_policies():
    f = open("working_hypotheses.txt", "r")
    hypotheses = eval(f.read())
    hypo_policies = {}
    for theta in hypotheses:
        if theta != "center" and theta != "anywhere" and theta != "crash":
            hypo_policies[theta] = gen_traj(15, theta = theta)
            print("Done {}".format(theta))
    pickle.dump(hypo_policies, open("working_hypo_policies.pkl", "wb"))

### Code to find failed hypotheses ###
def find_failed():
    policies = pickle.load(open("working_hypo_policies.pkl", "rb"))
    failed = []
    for k in policies:
        if len(policies[k]) == 0:
            failed.append(k)
    print(failed)

### Code to aggregate all hypothesis policies ###
def aggregate_policies():
    all_hypo_policies = pickle.load(open("hypothesis_policies.pkl", "rb"))
    working_policies = pickle.load(open("working_hypo_policies.pkl", "rb"))
    all_hypo_policies.update(working_policies)
    pickle.dump(all_hypo_policies, open("hypothesis_policies.pkl", "wb"))
    print("Done aggregating {} policies".format(len(all_hypo_policies)))

if __name__ == "__main__":
    # generate_hypotheses()
    # train_DQNs()
    # generate_policies()
    # find_failed()
    # aggregate_policies()
    print()