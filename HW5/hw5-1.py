"""
2022ML homework5-1 created by Pei Hsuan Tsai.
Gaussian Process
    Implement the Gaussian Process and Visualize the result.
    Task 1: Apply Gaussian Process Regression to predict the distribution of f.
    Task 2: Optimize the Kernel Parameters by minimizing negative marginal log-likelihood.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def InputData():
    'Read the data from the file.\nOutput: data X and data Y'
    # Matrix A represents 34 data entries, each data point (x, y)
    dataX = []
    dataY = []
    with open('./data/input.data', 'r') as f:
        L = f.readlines()
        for line in L:
            point = line.split()    # remove leading space of the data line
            dataX.append(float(point[0]))
            dataY.append(float(point[1]))
    return dataX, dataY


def Kernel(xn, xm, all_var, length, a):
    'Use a rational quadratic kernel to compute similarities between different points.\nOutput : kernel value'
    # Rational quadratic kernel(Xn, Xm) = var * [1 + (Xn-Xm)^2 / (2*a*l^2)]^-a
    diff = np.abs(xn-xm) ** 2
    tmp = 1 + diff / (2*a*(length**2))
    kernel = all_var * (tmp**(-a))
    return kernel


def CovarianceMatrix(x, all_var, length, a, b):
    'Compute the covariance matrix by the kernel.\nOutput : covariance matrix C'
    N = len(x)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                C[i][j] = Kernel(x[i], x[j], all_var, length, a) + (1/b)
            else:
                C[i][j] = Kernel(x[i], x[j], all_var, length, a)
    return C


def ConditionalDistribution(x, y, c, all_var, length, a, b):
    'Compute the mean and variance of the conditional distribution.\nOutput : mean and vatrance'
    N = len(x)
    sample_num = 1000
    Y = np.array(y).reshape((N, 1))
    test_x = np.linspace(-60, 60, sample_num)
    mean = np.zeros(sample_num)
    var = np.zeros(sample_num)
    k_test = np.zeros(sample_num)
    for i in range(sample_num):
        K = np.zeros((N, 1))
        for k in range(N):
            K[k][0] = Kernel(x[k], test_x[i], all_var, length, a)
        # k* = kernel(x*, x*) + b^-1
        k_test[i] = Kernel(test_x[i], test_x[i], all_var, length, a) + (1/b)
        mul = np.matmul(K.T, np.linalg.inv(c))
        # mean = kernel(x, x*)T * C^-1 * y
        mean[i] = np.matmul(mul, Y)
        # variance = k* - kernel(x, x*)T * C^-1, kernel(x, x*)
        var[i] = k_test[i] - np.matmul(mul, K)
    return mean, var


def PredictionResult(x, y, mean, var):
    'Visualize the prediction result.'
    sample_num = 1000
    test_x = np.linspace(-60, 60, sample_num)
    # Draw a line to represent mean of f in range [-60, 60]
    plt.plot(test_x, mean, color='black')
    # Mark the 95% confidence interval of f -> confidence level = 1.96
    interval = 1.96 * np.sqrt(var)
    plt.fill_between(test_x, mean+interval, mean-interval, color='yellow')
    # Show all training data points
    plt.scatter(x, y, color='blue')
    plt.show()


def GaussianProcess(X, Y, all_var, lengthscale, a, b):
    'Apply Gaussian Process Regression to predict the distribution.'
    # Calculate the covariance matrix C by the kernel
    C = CovarianceMatrix(X, all_var, lengthscale, a, b)
    # Compute the conditional distribution ~ N(mean, variance)
    mean, var = ConditionalDistribution(X, Y, C, all_var, lengthscale, a, b)
    # Use the conditional distribution to get prediction
    PredictionResult(X, Y, mean, var)


# fun(x, *args) => x: array of variables to minimize (all_var, lengthscale, a), *args: tuple of fixed parameters (x, y, b)
def LogLikelihood(theta, *args):
    'Calculate negative marginal log-likelihood.\nOutput : value of likelihood'
    N = len(args[0][0])
    y = np.array(args[0][1]).reshape((N, 1))
    # negative marginal log-likelihood = (1/2)*log(|C|) + (1/2)*yT*C^-1*y + (N/2)*log(2*pi)
    C = CovarianceMatrix(args[0][0], theta[0], theta[1], theta[2], args[0][2])
    likelihood = (N/2) * np.log(2 * np.pi)
    likelihood += (1/2) * np.matmul(np.matmul(y.T, np.linalg.inv(C)), y)
    likelihood += (1/2) * np.log(np.linalg.det(C))
    return float(likelihood)


if __name__ == '__main__':
    X, Y = InputData()
    all_var = 1 # overall vatiance for rational quadratic kernel
    lengthscale = 1 # lengthscale for rational quadratic kernel
    a = 1   # scale-mixture for rational quadratic kernel (a>0)
    b = 5   # beta for Gaussian

    # Task 1
    GaussianProcess(X, Y, all_var, lengthscale, a, b)

    # Task 2
    # Optimize the Kernel Parameters by minimizing negative marginal log-likelihood
    res = minimize(LogLikelihood, x0=[all_var, lengthscale, a], args=[X, Y, b])
    GaussianProcess(X, Y, res.x[0], res.x[1], res.x[2], b)