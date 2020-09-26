import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os


"""
 * train a behavior cloned policy from (s,a) pairs
 * Dylan Losey, September 2020
"""


# collect dataset
class MotionData(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


# lightweight multilayer perceptron
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        # state is joint position, size 7
        # action is change in joint position, size 7
        self.fc1 = nn.Linear(7,14)
        self.fc2 = nn.Linear(14,14)
        self.fc3 = nn.Linear(14,7)
        self.loss_func = nn.MSELoss()

    def predict(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc3(h2)

    def forward(self, x):
        s = x[:, 0:7]
        a = x[:, 7:14]
        ahat = self.predict(s)
        loss = self.loss(a, ahat)
        return loss

    def loss(self, a_target, a_predicted):
        return self.loss_func(a_target, a_predicted)


# train and save model
def main():

    folder = "demos"
    savename = "models/bc-test.pt"

    dataset = []
    for filename in os.listdir(folder):
        xi = pickle.load(open(folder + "/" + filename, "rb"))
        for item in xi:
            dataset.append(item)
    print("my dataset has this many (s,a) pairs: ", len(dataset))


    # hyperparameters to tune
    EPOCH = 1000
    BATCH_SIZE_TRAIN = 100
    LR = 0.01
    LR_STEP_SIZE = 300
    LR_GAMMA = 0.1

    model = MLP()
    train_data = MotionData(dataset)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
