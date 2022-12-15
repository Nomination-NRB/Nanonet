import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import util
import dataset
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainData, trainLabel = util.utilmain()

trainData = trainData.to(float).to(device)
trainLabel = trainLabel.to(float).to(device)

dataset = dataset.MyDataset(trainData, trainLabel)
dataloder = DataLoader(dataset, batch_size=16, shuffle=True)

model = model.Net().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    losslist = []
    print('training on ', device)
    for epo in range(epoch):
        train_loss = 0
        for i, data in enumerate(dataloder):
            inputs, labels = data
            inputs = inputs.to(float)
            labels = labels.to(float)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(float)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        losslist.append(train_loss/len(dataloder))
    plt.plot(np.arange(len(losslist)), losslist, label="train loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    print('Finished Training')


if __name__ == '__main__':
    train(100)
