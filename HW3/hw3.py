"""
2022ML homework3 created by Pei Hsuan Tsai.
1. Random Data Generator
    a. Univariate gaussian data generator
    b. Polynomial basis linear model data generator
2. Sequential Estimator (data from 1.a)
3. Baysian Linear regression (data from 1.b)
"""


import numpy as np
import matplotlib.pyplot as plt
import math


def UnivariatGaussian(m, s):
    'Univariate gaussian data generator. Input mean m, variance s.\nOutput : a data from N(m,s)'
    # By Marsaglia polar method (modification of the Bix-Muller, doesn't need sin,cos)
    # Choose random U and V untill 0 < s < 1
    S = 0
    while S >= 1 or S <= 0 :
        # U, V independent random on (-1, 1)
        U = np.random.uniform(-1, 1)
        V = np.random.uniform(-1, 1)
        S = U ** 2 + V ** 2        
    # Calculate x = U*sqrt(-2*ln(S)/S), y = V*sqrt(-2*ln(S)/S)
    S = math.sqrt(-2 * math.log(S) / S)
    x = U * S
    # data = x * standard_deviation + mean
    return x * math.sqrt(s) + m


def PolynomialBasis(n, a, w):
    'Polynomial basis linear model data generator. Input n(basis num.), a, w.\nOutput : a point (x, y)'
    # x = uniformly distributed on (-1, 1)
    x = np.random.uniform(-1, 1)
    # e ~ N(0, a)
    e = UnivariatGaussian(0, a)
    # y = wT * x + e
    y = e
    for i in range(n):
        y += w[i] * (x ** i)
    return x, y


def SequentialEstimator(m, s):
    'Sequential estimate the mean and variance. Input as UnivariatGaussian.'
    print(f"Data point source function: N({m}, {s})")
    # Call UnivariatGaussian to get a new data point from N(m, s)
    data = UnivariatGaussian(m, s)
    print("Add data point:", data)
    # Use sequential estimation to find the current estimates to m and s
    # Repeat steps above until the estimates converge
    # print("Mean =", m, "Variance = ", s)
    return


def BaysianRegression(b, n, a, w):
    'Baysian Linear regression. Input for initial prior and for PolynomialBasis.'
    data = PolynomialBasis(n, a, w)
    print(f"Add data point({data[0]}, {data[1]}):")
    # print("Posterior mean:")
    # print("Posterior variance:")
    # print("Predictive distribution ~ N(", mean, ",", var, ")")
    return


def Visualization():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_title("Ground truth")
    ax2.set_title("Predict result")
    ax3.set_title("After 10 incomes")
    ax4.set_title("After 50 incomes")
    # ax1.plot(xf, y1)
    # ax2.plot(xf, y2)
    plt.show()
    return


if __name__ == '__main__':
    m = 3
    s = 5
    b = 1
    n = 4
    a = 1
    w = [1, 2, 3, 4]
    SequentialEstimator(m, s)
    BaysianRegression(b, n, a, w)
    Visualization()