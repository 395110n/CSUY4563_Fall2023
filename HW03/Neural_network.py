import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

trainingdata = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=True,download=True,transform=torchvision.transforms.ToTensor())
testdata = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=False,download=True,transform=torchvision.transforms.ToTensor())

device=torch.device("cuda:0")

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784, out_features=200, device=device), nn.ReLU(),
    nn.Linear(in_features=200, out_features=200, device=device), nn.ReLU(),
    nn.Linear(in_features=200, out_features=10, device=device)
    )

def train(net, trainingdata, testdata, batch_size, lr, epochs, device):
    trainDataLoader = torch.utils.data.DataLoader(trainingdata, batch_size=batch_size, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
    updater = torch.optim.SGD(net.parameters(), lr = lr)
    all_train_loss = []
    all_test_loss = []
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in trainDataLoader:
            X, y = X.to(device), y.to(device)
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            updater.step()
            total_loss += l.item()
        avg_loss = total_loss/len(trainDataLoader)
        print("epoch: ", epoch, " average loss: ", avg_loss)
        all_train_loss.append(avg_loss)
        
        total_loss = 0
        with torch.no_grad():
            for X_test, y_test in testDataLoader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_hat_test = net(X_test)
                l = loss(y_hat_test, y_test)
                total_loss += l.item()
            avg_loss = total_loss / len(testDataLoader)
            all_test_loss.append(avg_loss)

    plt.plot(range(epochs), all_train_loss, color = "blue", label = "train", marker = None)
    plt.plot(range(epochs), all_test_loss, color = "red", label = "test",  marker = None)
    plt.show()
  
# train(net, trainingdata, testdata, batch_size=64, lr=0.1, epochs=30, device=device)
train(net, trainingdata, testdata, batch_size=16, lr=0.5, epochs=30, device=device)
