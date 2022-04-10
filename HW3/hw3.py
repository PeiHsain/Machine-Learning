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


def ScaleIMatrix(b, n):
    'Scale the n*n identity matrix by b.\nOutput : matrix b^(-1)*I'
    M = np.zeros([n, n])
    for i in range(n):
        M[i][i] = 1 / b
    return M


def XMatrix(x, n):
    'Create a N*1 matrix for data x.\nOutput : matrix of x'
    M = np.zeros([1, n])
    for i in range(n):
        M[0][i] = x ** i
    return M


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
    # Use Welford's online algorithm, add one x each time to update mean and variance
    print(f"Data point source function: N({m}, {s})")
    mean = 0
    variance = 0
    M2 = 0
    n = 0
    converge = 0.0001
    count = 0

    # Repeat steps above until the estimates converge or get times limited
    while count < 10 and n < 18000:
        old_m = mean
        old_var = variance
        # Call UnivariatGaussian to get a new data point from N(m, s)
        data = UnivariatGaussian(m, s)
        n += 1
        print("Add data point:", data)

        # Use sequential estimation to find the current estimates to m and s
        tmp = data - mean
        # mean(n) = mean(n-1) + (xn - mean(n-1))/n
        mean += (tmp / n)
        # M2(n) = M2(n-1) + (xn - mean(n-1))*(xn - mean(n))
        M2 += (tmp * (data - mean))
        # unbiased variance(n) = M2(n) / (n - 1), n > 1
        if n > 1:
            variance = (M2 / (n-1))
        print(f"Mean = {mean}, Variance = {variance}")

        # Repeat steps above until the estimates converge, converge range = 0.0001 for 10 times
        if abs(old_m - mean) <= converge and abs(old_var - variance) <= converge:
            count += 1
        else:
            count == 0
    return


def BaysianRegression(b, n, a, w):
    'Baysian Linear regression. Input for initial prior and for PolynomialBasis.'
    # Use MAP and online learing, add one data each time to update prior, and calculate posterior and distribution
    a = 1/a
    converge = 0.001
    count = 0
    times = 0
    data_x = []
    data_y = []
    m_for_visual = []
    var_for_visual = []
    # initial prior ~ N(0, b^(-1)I)
    mean = np.zeros([n, 1])
    variance = ScaleIMatrix(b, n)
    
    # Repeat steps until the posterior probability converges or get times limited
    while count < 10 and times < 3000:
        conv_flag = True
        old_m = mean
        old_var = variance
        # Call PolynomialBasis to generate one data point ~ N(y, a^(-1))
        data = PolynomialBasis(n, 1/a, w)
        print(f"Add data point ({data[0]}, {data[1]}):")
        data_x.append(data[0])
        data_y.append(data[1])
        X = XMatrix(data[0], n)
        Y = np.array([[data[1]]])
        times += 1

        # Update the prior ~ N(mean, var), and calculate the parameters of predictive distribution
        # mega = a*XT*X + var^(-1)
        mega = a * np.matmul(X.T, X) + np.linalg.inv(variance)
        # posterior mean = mega^(-1) * (a*XT*Y + var^(-1)*mean)
        tmp = a * np.matmul(X.T, Y) # a*XT*Y
        tmp2 = np.matmul(np.linalg.inv(variance), mean)  # var^(-1)*mean
        mean = np.matmul(np.linalg.inv(mega), (tmp + tmp2))
        print("Posterior mean:", mean)
        # posterior variance = mega^(-1)
        variance = np.linalg.inv(mega)
        print("Posterior variance:", variance)
        # predictive distribution = N(X*mean, 1/a + X*var*XT)
        update_m = float(np.matmul(X, mean))
        update_var = float((1/a) + np.matmul(np.matmul(X, variance), X.T))
        print(f"Predictive distribution ~ N({update_m}, {update_var})")
        print()

        # Repeat steps above until the posterior probability converges, converge range = 0.001 for 10 times, 
        for i in range(n):
            for j in range(n):
                if abs(old_var[i][j] - variance[i][j]) > converge:
                    conv_flag = False
            if abs(old_m[i][0] - mean[i][0]) > converge:
                conv_flag = False
        if conv_flag == False:
            count = 0
        else:
            count += 1

        # Keep the mean and variance on 10 times, 50 times
        if times == 10 or times == 50:
            m_for_visual.append(mean)
            var_for_visual.append(variance)

    # keep the final mean and variance
    m_for_visual.append(mean)
    var_for_visual.append(variance)
    # After probability converged, do the visualization
    Visualization(data_x, data_y, n, a, w, m_for_visual, var_for_visual)
    return


def Visualization(dataX, dataY, n, a, w, m, var):
    'Visualization of ground truth and prediction.'
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sampleNum = 500
    x = np.linspace(-2, 2, sampleNum)
    y = np.zeros(sampleNum)
    y_up = np.zeros(sampleNum)
    y_down = np.zeros(sampleNum)

    # Figure 1: ground truth, y = wT*x, y = wT*x +- variance
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-20, 25)
    ax1.set_title("Ground truth")
    for i in range(sampleNum):
        tmpX = XMatrix(x[i], n)
        y[i] = float(np.matmul(tmpX, w))
        y_up[i] = y[i] + 1/a
        y_down[i] = y[i] - 1/a
    ax1.plot(x, y, color='black')
    ax1.plot(x, y_up, color='red')
    ax1.plot(x, y_down, color='red')

    # Figure 2: Final predict result, y = x*mean, y = x*mean +- variance
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-20, 25)
    ax2.set_title("Predict result")
    ax2.scatter(dataX, dataY)
    for i in range(sampleNum):
        tmpX = XMatrix(x[i], n)
        v = float((1/a) + np.matmul(np.matmul(tmpX, var[2]), tmpX.T))
        y[i] = float(np.matmul(tmpX, m[2]))
        y_up[i] = y[i] + v
        y_down[i] = y[i] - v
    ax2.plot(x, y, color='black')
    ax2.plot(x, y_up, color='red')
    ax2.plot(x, y_down, color='red')

    # Figure 3: At the time that have seen 10 data points
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-20, 25)
    ax3.set_title("After 10 incomes")
    ax3.scatter(dataX[:10], dataY[:10])
    for i in range(sampleNum):
        tmpX = XMatrix(x[i], n)
        v = float((1/a) + np.matmul(np.matmul(tmpX, var[0]), tmpX.T))
        y[i] = float(np.matmul(tmpX, m[0]))
        y_up[i] = y[i] + v
        y_down[i] = y[i] - v
    ax3.plot(x, y, color='black')
    ax3.plot(x, y_up, color='red')
    ax3.plot(x, y_down, color='red')

    # Figure 4: At the time that have seen 50 data points
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-20, 25)
    ax4.set_title("After 50 incomes")
    ax4.scatter(dataX[:50], dataY[:50])
    for i in range(sampleNum):
        tmpX = XMatrix(x[i], n)
        v = float((1/a) + np.matmul(np.matmul(tmpX, var[1]), tmpX.T))
        y[i] = float(np.matmul(tmpX, m[1]))
        y_up[i] = y[i] + v
        y_down[i] = y[i] - v
    ax4.plot(x, y, color='black')
    ax4.plot(x, y_up, color='red')
    ax4.plot(x, y_down, color='red')
    
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
    # BaysianRegression(b, n, a, w)