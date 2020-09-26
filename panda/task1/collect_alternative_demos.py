from env import Task
import numpy as np
import time
import pickle
from train_bc_human import MLP
import torch


class Model(object):

    def __init__(self, filename):
        self.model = MLP()
        model_dict = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def predict(self, state):
        s = state.tolist()
        s_tensor = torch.FloatTensor(s)
        a_tensor = self.model.predict(s_tensor)
        return a_tensor.detach().numpy()



def get_noise(model, episodes):
    hold_random = 100
    variance_random = 0.05
    env = Task()
    dataset = []
    cycle = 1000
    max_count = 10001
    timesteps = np.linspace(0, 5, max_count)
    for episode in range(episodes):
        state = env.reset()
        count = 0
        xi = []
        qdot_prev = np.asarray([0] * 7)
        while count < max_count:
            curr_time = timesteps[count]
            if count % 1000 == 0:
                xi.append(state["joint_position"][0:7].tolist())
            s = state["joint_position"][0:7]
            if count % cycle == 0:
                count_random = hold_random
                qdot = np.random.normal(0, variance_random, 7)
            if count_random > 1:
                count_random -= 1
            else:
                qdot = 0.2 * model.predict(state["joint_position"][0:7])
            next_state, reward, done, info = env.step(qdot)
            state = next_state
            qdot_prev = qdot
            count += 1
        dataset.append(xi)
    env.close()
    return dataset



def get_counterfactual(model, episodes):
    hold_random = 100
    variance_random = 0.05
    env = Task()
    dataset = []
    cycle = 1000
    max_count = 10001
    timesteps = np.linspace(0, 5, max_count)
    for episode in range(episodes):
        state = env.reset()
        count = 0
        xi = []
        qdot_prev = np.asarray([0] * 7)
        action_scale = np.random.random()
        while count < max_count:
            curr_time = timesteps[count]
            if count % 1000 == 0:
                xi.append(state["joint_position"][0:7].tolist())
            s = state["joint_position"][0:7]
            if count % cycle == 0:
                count_random = hold_random
                qdot = np.random.normal(0, variance_random, 7)
            if count_random > 1:
                count_random -= 1
            else:
                qdot = 0.2 * model.predict(state["joint_position"][0:7]) * action_scale
            next_state, reward, done, info = env.step(qdot)
            state = next_state
            qdot_prev = qdot
            count += 1
        dataset.append(xi)
    env.close()
    return dataset


def main():

    filename = "models/bc-test.pt"
    model = Model(filename)

    noisies = get_noise(model, 50)
    counterfactuals = get_counterfactual(model, 50)

    pickle.dump( noisies, open( "choices/noisy.pkl", "wb" ) )
    pickle.dump( counterfactuals, open( "choices/counterfactual.pkl", "wb" ) )


if __name__ == "__main__":
    main()
