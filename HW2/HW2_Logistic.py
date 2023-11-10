import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

performance = {}
training_cost = {}
prediction_cost = {}

X_train = train_images.reshape(train_images.shape[0], -1)
y_train = train_labels

X_test = test_images.reshape(test_images.shape[0], -1)
y_test = test_labels

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for index in range(2, 10):
    T1 = time.time()
    Logistic = LogisticRegression(C=index, tol = (10 ** (-index)))
    Logistic.fit(X_train, y_train)
    T2 = time.time()

    test_score = Logistic.score(X_test, y_test)
    T3 = time.time()
    performance[test_score] = index
    training_cost[test_score] = (T2 - T1)
    prediction_cost[test_score] = (T3 - T2)
    print("Index: ", index,  " Testing Score: ", test_score)

test_scores = list(performance.keys())
test_scores.sort()
ranked_performance = {performance[score] : score for score in test_scores}
ranked_training_cost = {training_cost[score] : score for score in test_scores}
ranked_prediction_cost = {prediction_cost[score] : score for score in test_scores}
print(ranked_performance)
print(ranked_training_cost)
print(ranked_prediction_cost)