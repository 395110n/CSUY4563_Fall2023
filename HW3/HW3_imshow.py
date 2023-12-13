import torch
import torchvision
import matplotlib.pyplot as plt
     
trainingdata = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=True,download=True,transform=torchvision.transforms.ToTensor())
testdata = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=False,download=True,transform=torchvision.transforms.ToTensor())

trainDataLoader = torch.utils.data.DataLoader(trainingdata, batch_size=64, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=False)

images, labels = next(iter(trainDataLoader))
labels_list = list(labels)

plt.figure(figsize=(10,4))
for index in range(10):
    plt.subplot(2,5,index+1)
    plt.imshow(images[labels_list.index(index)].squeeze().numpy())

plt.show()

