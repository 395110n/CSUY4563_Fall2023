import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


indexes = {num : [] for num in range(10)}
for key in indexes.keys():
    index = 0
    for label in train_labels:
        if label == key and index not in indexes[key]:
            indexes[key].append(index)
        index += 1
        if len(indexes[key]) == 10:
            break
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
images = {numClass: [train_images[index] for index in indexes[numClass]] for numClass in indexes.keys()}
for i in range(10):
    for j in range(10):
        img = images[i][j]
        ax = axes[i, j]
        ax.imshow(img) 
        ax.axis("off")

plt.show()