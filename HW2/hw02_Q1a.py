from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

samples = np.array([[0, 1], [2, 3], [4, 4], [2, 0], [5, 2], [6, 3]])

labels = np.array([0, 0, 0, 1, 1, 1])

X = samples[:, 0]
y = samples[:, 1]

model = LogisticRegression()
model.fit(samples, labels)

x_min, x_max = samples[:, 0].min() - 1, samples[:, 0].max() + 1
coef = model.coef_
intercept = model.intercept_
xx = np.linspace(x_min, x_max, 100)
yy = (-1 / coef[0][1]) * (coef[0][0] * xx + intercept)

# 绘制决策边界
plt.plot(xx, yy)
plt.scatter(X[:3], y[:3], c="blue")
plt.scatter(X[3:], y[3:], c="red")
plt.show()

