import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# random data
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)


# plotting x and y
plt.scatter(x, y)


theta_best = np.linalg.inv(x.T.dot(X)).dot(x.T).dot(y)

def cal_cost(theta, x, y):
    m = len(y)
    predictions = x.dot(theta)
    cost = (1/2 * m) * np.sum(np.square(predictions-y))

def gradient_decent(x, y, theta, learning_rate=0.1, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        predictions = np.dot(x, theta)
        theta = theta - (1/m) * learning_rate * (x.T.dot((predictions - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, x, y)

    return theta, cost_history, theta_history

l_r = 0.1
n_iter = 100

theta = np.random.randn(2, 1)

x_b = np.c_[np.ones((len(x), 1), x]
theta, cost_history, theta_history = gradient_decent(x, y, theta, lr, n_iter)