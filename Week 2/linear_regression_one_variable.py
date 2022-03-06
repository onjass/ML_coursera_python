# Author : Mike Jason (onjass on GitHub)
# About : Programming Assignment: Linear Regression - https://www.coursera.org/learn/machine-learning/programming/8f3qT/linear-regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    data = pd.read_csv("ex1data1.txt", sep=",", header=None).to_numpy()
    x = data[:,0]
    y = data[:,1]
    return x, y

def plot_data(x, y):
    plt.plot(x, y, 'rx')
    plt.ylabel("Profit in $10.000s")
    plt.xlabel("Population of City in 10,000s")
    plt.show()

def computeCost(x, y, m, theta):
    h_x = np.sum(x * np.transpose(theta), axis=1)
    J = (1/(2 * m)) * sum(np.square(h_x - y))
    return J

def gradientDescent(x, y, m, theta, alpha, iterations):
    J_history = np.zeros((iterations, 1))
    for iter in range(0, iterations):
        error = np.sum(np.transpose(theta) * x, axis=1) - y
        theta = theta - ((alpha / m) * np.transpose(x) * error)
        J_history[iter] = computeCost(x, y, m, theta)
    return J_history

if __name__ == "__main__":
    ############ PART 1 : Plotting Data ############
    print("Plotting Data ...")
    x, y = load_data()
    m = len(y) # Number of training examples
    # plot_data(x, y)

    ############ PART 2 : Cost and Gradient Descent ############
    x = np.transpose(np.array([np.ones(m), x]))
    theta = np.zeros((2,1))
    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print("Testing the cost function ...")
    J = computeCost(x, y, m, theta) # with theta set to 0
    print("With theta = [0 ; 0], the cost computed = {}".format(J))
    J = computeCost(x, y, m, np.array([-1, 2]))
    print("With theta = [-1 ; 2], the cost computed = {}".format(J))
