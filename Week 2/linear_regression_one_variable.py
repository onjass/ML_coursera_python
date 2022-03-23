# Author : Mike Jason (onjass on GitHub)
# About : Programming Assignment: Linear Regression - https://www.coursera.org/learn/machine-learning/programming/8f3qT/linear-regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    data = pd.read_csv("ex1data1.txt", sep=",", header=None).to_numpy()
    x, y = data[:,0], data[:,1]
    return x, y

def plot_data(x, y):
    fig = plt.figure()
    plt.plot(x, y, 'ro')
    plt.ylabel("Profit in $10.000s")
    plt.xlabel("Population of City in 10,000s")

def computeCost(x, y, m, theta):
    h_x = np.dot(x, theta)
    J = (1/(2 * m)) * np.sum(np.square(h_x - y))
    return J

def gradientDescent(x, y, m, theta, alpha, iterations):
    theta = theta.copy()
    J_history = []
    for iter in range(iterations):
        theta = theta - (alpha / m) * (np.dot(x, theta) - y).dot(x)
        J_history.append(computeCost(x, y, m, theta))
    return theta, J_history

if __name__ == "__main__":
    ############ PART 1 : Plotting Data ############
    print("Plotting Data ...")
    x, y = load_data()
    m = y.size # Number of training examples
    plot_data(x, y)
    plt.show()

    ############ PART 2 : Cost and Gradient Descent ############
    x = np.stack([np.ones(m), x], axis=1)
    theta = np.zeros(2)
    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print("Testing the cost function ...")
    J = computeCost(x, y, m, theta) # with theta set to 0
    print("With theta = [0 ; 0], the cost computed = {}".format(J))
    J = computeCost(x, y, m, np.array([-1, 2]))
    print("With theta = [-1 ; 2], the cost computed = {}".format(J))

    print("Running Gradient Descent ...")
    theta, J_history = gradientDescent(x, y, m, theta, alpha, iterations)
    print("Theta found by gradient descent: {:.4f}, {:.4f}".format(*theta))
    # Plotting training data with linear fit found
    plot_data(x[:,1], y)
    plt.plot(x[:,1], np.dot(x, theta), '-')
    plt.legend(["Training Data", "Linear Regression"])
    plt.show()

    # Predict values for population sizes 35k and 70k
    predict1 = np.dot([1, 3.5], theta)
    print("For population = 35k, we predict a profit of {:.2f}".format(predict1*10000))
    predict1 = np.dot([1, 7], theta)
    print("For population = 35k, we predict a profit of {:.2f}".format(predict1*10000))
