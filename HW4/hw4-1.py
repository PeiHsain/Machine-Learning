"""
2022ML homework4-1 created by Pei Hsuan Tsai.
Logistic regression
    Use Logistic regression to separate D1 and D2.
    Implement both Newton's and steepest gradient descent method during optimization.
"""


import numpy as np
import matplotlib.pyplot as plt
import math


def XMatrix(x, n):
    'Create a x.size*N matrix for data x.\nOutput : matrix of x'
    M = np.ones([len(x), n])
    for i in range(len(x)):
        for j in range(1, n):
            M[i][j] = x[i][j-1]
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


def LogicFunc(w, x):
    'Calculate the logic function with w and x.\nOutput : calculated value'
    logic = np.zeros([len(x), 1])
    # Logic function = 1 / (1 + exp^(-wT*x))
    for i in range(len(x)):
        tmp = np.matmul(x[i], w)
        # Deal with out of range
        try:
            logic[i] = 1 / (1 + math.exp(-tmp))
        except OverflowError:
            logic[i] = 0
    return logic


def Diagonal(w, x):
    'Calculate the n*n diagonal matrix.\nOutput : diagonal matrix'
    D = np.zeros([len(x), len(x)])
    # diagonal matrix = exp(-WT*X) / (1+exp(-WT*X))^2
    for i in range(len(x)):
        tmp = np.matmul(x[i], w)
        # Deal with out of range
        try:
            D[i][i] = math.exp(-tmp) / ((1 + math.exp(-tmp)) ** 2)
        except OverflowError:
            D[i][i] = 0
    return D


def Prediction(w, x):
    'Use model w to classifier(y = 0 or 1).\nOutput : the prediction'
    # activte > 0.5 -> y = 1(class 2), activate <= 0.5 -> y = 0(class 1)
    y_pred = np.zeros([len(x), 1])
    pred_value = LogicFunc(w, x)
    for i in range(len(x)):
        if pred_value[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


def ConfusionMat(y_label, y_pred):
    'Create and print the confusion matrix w.\nOutput : the confusion matrix'
    print('Confusion Matrix:')
    # TP = [0][0], FN = [0][1], FP = [1][0], TN = [1][1]
    mat = np.zeros([2, 2])
    # Calculate TP, FN, FP, TN
    for i in range(len(y_pred)):
        if y_pred[i] == y_label[i]:
            if y_label[i] == 0:
                mat[0][0] += 1  # TP
            else:
                mat[1][1] += 1  # TN
        else:
            if y_label[i] == 0:
                mat[0][1] += 1  # FN
            else:
                mat[1][0] += 1  # FP
    # Print matrix
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print(f'Is cluster 1\t\t{mat[0][0]}\t\t\t{mat[0][1]}')
    print(f'Is cluster 2\t\t{mat[1][0]}\t\t\t{mat[1][1]}')
    return mat


def Sensitivity(mat):
    'Calculate sensitivity of the logistic regression.\nOutput : sensitivity value'
    # sensitivity = TP / (TP + FN)->Yes
    if mat[0][0] + mat[0][1] != 0:
        sensitivity = mat[0][0] / (mat[0][0] + mat[0][1])
    else:
        sensitivity = 0
    print(f'Sensitivity (Successfully predict cluster 1): {sensitivity}')
    return


def Specificity(mat):
    'Calculate specificity of the logistic regression.\nOutput : specificity value'
    # specificity = TN / (TN + FP)->No
    if mat[1][0] + mat[1][1] != 0:
        specificity = mat[1][1] / (mat[1][0] + mat[1][1])
    else:
        specificity = 0
    print(f'Specificity (Successfully predict cluster 2): {specificity}')
    return

def NewtonMethod(w, lr, x, y):
    'Implement Newton method to optimize logistic regression.'
    # Initialize
    x_matrix = XMatrix(x, 3)
    conv_count = 0
    conv_time = 0
    error = 0.001

    # When W is converge or iteration time meets converge_limit time, stop the loop
    while conv_count < 7 and conv_time < 10000 :
        old_w = np.copy(w)
        conv_flag = True
        conv_time += 1
        # Calculating the activate function (logic function) = 1 / (1 + exp(-WT*X))
        activate = LogicFunc(w, x_matrix)
        # Calculating the gradient = XT*(Y - activate)
        gradient = np.matmul(x_matrix.T, (y-activate))
        # Calculating the Hession = XT * D * X, D = exp(-WT*X) / (1+exp(-WT*X))^2
        D = Diagonal(w, x_matrix)
        H = np.matmul(np.matmul(x_matrix.T, D), x_matrix)
        # Updating the W by the Hession_inverse * gradients
        # If H is non-invertible (singular) -> gradient descent
        if np.linalg.det(H) == 0:
            # W(n+1) = W(n) + lr * gradient
            w += lr * gradient
        else:
            # W(n+1) = W(n) + lr * H^(-1) * gradient
            w += lr * np.matmul(np.linalg.inv(H), gradient)
        # Check converge or not -> difference of w <= error (converge flag = ture)
        for i in range(len(w)):
            if abs(old_w[i] - w[i]) > error:
                conv_flag = False
        if conv_flag == True:
            conv_count += 1
        else:
            conv_count = 0
    # Print W
    print(f'w:\n{w}')

    # Classifier
    y_pred = Prediction(w, x_matrix)
    # Print the confusion matrix
    confusion = ConfusionMat(y, y_pred)
    # Calculate sensitivity and specificity
    Sensitivity(confusion)
    Specificity(confusion)
    return y_pred


def GradientDescent(w, lr, x, y):
    'Implement steepest gradient descent to optimize logistic regression. "lr" is learning rate.\nOutput : prediction'
    # Initialize
    x_matrix = XMatrix(x, 3)
    conv_count = 0
    conv_time = 0
    error = 0.001

    # When W is converge or iteration time meets converge_limit time, stop the loop
    while conv_count < 7 and conv_time < 10000 :
        old_w = np.copy(w)
        conv_flag = True
        conv_time += 1
        # Calculating the activate function (logic function) = 1 / (1 + exp(-WT*X))
        activate = LogicFunc(w, x_matrix)
        # Updating W by the learning rate * gradients
        # W(n+1) = W(n) + lr * XT*(Y - activate) = W(n) + lr * gradient, lr = learing rate
        w += lr * np.matmul(x_matrix.T, (y-activate))
        # Check converge or not -> difference of w <= error (converge flag = ture)
        for i in range(len(w)):
            if abs(old_w[i] - w[i]) > error:
                conv_flag = False
        if conv_flag == True:
            conv_count += 1
        else:
            conv_count = 0
    # Print W
    print(f'w:\n{w}')

    # Classifier
    y_pred = Prediction(w, x_matrix)
    # Print the confusion matrix
    confusion = ConfusionMat(y, y_pred)
    # Calculate sensitivity and specificity
    Sensitivity(confusion)
    Specificity(confusion)
    return y_pred


def LogisticRegression(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    'Use Logistic regression to separate D1 and D2. Implement both Newton and gradient descent during optimization.'
    # Generate n data point D1(x, y) = x1~N(mx1, vx1) x2~N(my1, vy1), y = 0
    # Generate n data point D2(x, y) = x1~N(mx2, vx2) x2~N(my2, vy2), y = 1
    D1 = np.zeros([n, 2])
    D2 = np.zeros([n, 2])
    for i in range(n):
        D1[i][0] = UnivariatGaussian(mx1, vx1)
        D1[i][1] = UnivariatGaussian(my1, vy1)
        D2[i][0] = UnivariatGaussian(mx2, vx2)
        D2[i][1] = UnivariatGaussian(my2, vy2)
    x = np.concatenate([D1, D2])
    y = np.concatenate([np.zeros([n, 1]), np.ones([n, 1])])

    # Seeptest gradient descent
    print("Gradient descent:")
    w = np.zeros([3, 1])
    GlearningR = 0.5
    y_GPred = GradientDescent(w, GlearningR, x, y)
    print("------------------------------------")
    # Newton's method
    print("Newton's method:")
    w = np.zeros([3, 1])
    NlearningR = 0.1
    y_NPred = NewtonMethod(w, NlearningR, x, y)

    # Visualization
    Visualization(D1, D2, x, y_GPred, y_NPred)   
    return


def Visualization(D1, D2,x , y_GPred, y_NPred):
    'Visualization of ground truth and prediction.'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Figure 1: ground truth
    ax1.set_title("Ground truth")
    ax1.scatter(D1[:, 0], D1[:, 1], color='blue')
    ax1.scatter(D2[:, 0], D2[:, 1], color='red')

    # Figure 2: gradient descent
    ax2.set_title("Gradient descent")
    for i in range(len(x)):
        if y_GPred[i] == 0:
            ax2.scatter(x[i][0], x[i][1], color='blue')
        else:
            ax2.scatter(x[i][0], x[i][1], color='red')

    # Figure 3: Newton's method
    ax3.set_title("Newton's method")
    for i in range(len(x)):
        if y_NPred[i] == 0:
            ax3.scatter(x[i][0], x[i][1], color='blue')
        else:
            ax3.scatter(x[i][0], x[i][1], color='red')
    
    plt.show()
    return


if __name__ == '__main__':
    N = 50
    mx1 = 1
    my1 = 1
    mx2 = 10
    my2 = 10
    vx1 = 2
    vy1 = 2
    vx2 = 2
    vy2 = 2
    LogisticRegression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
