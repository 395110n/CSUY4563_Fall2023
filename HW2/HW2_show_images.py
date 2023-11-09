import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
indexes = []
for classNum in range(10):
    index = 0
    for label in train_labels:
        if label != classNum:
            index += 1
        else:
            indexes.append(index)
            break

images = [train_images[i] for i in indexes]
for i in range(10):
    img = images[i]
    ax = axes[i // 5, i % 5]
    ax.imshow(img)  

plt.show()