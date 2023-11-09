import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

performance = {}

X_train = train_images.reshape(train_images.shape[0], -1)
y_train = train_labels

X_test = test_images.reshape(test_images.shape[0], -1)
y_test = test_labels

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for index in range(2, 10):
    Logistic = LogisticRegression(C=index, tol = (10 ** (-index)))
    Logistic.fit(X_train, y_train)

    test_score = Logistic.score(X_test, y_test)
    performance[test_score] = index
    print("Index: ", index,  " Testing Score: ", test_score)

test_scores = list(performance.keys())
test_scores.sort()
ranked_performance = {performance[score] : score for score in test_scores}
print(ranked_performance)

