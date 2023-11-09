import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

X_train = train_images.reshape(train_images.shape[0], -1)
y_train = train_labels

X_test = test_images.reshape(test_images.shape[0], -1)
y_test = test_labels
performance = {}
for k in range(1, 21):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train, y_train)
    test_score = KNN.score(X_test, y_test)
    performance[test_score] = k
    print("k=",k, " Testing Score: ", test_score)

test_scores = list(performance.keys())
test_scores.sort()
ranked_performance = {performance[score] : score for score in test_scores}
print(ranked_performance)
