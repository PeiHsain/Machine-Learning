"""
2022ML homework5-2 created by Pei Hsuan Tsai.
SVM on MNIST
    Use SVM models to tackle classification on images of hand-written digits.
    Task 1: Use different kernel functions (linear, polynomial, and RBF kernels) and have comparison between their performance.
    Task 2: Please use C-SVC.
    Task 3: Use linear kernel + RBF kernel together (therefore a new kernel function) and compare its performance with respect to others.
"""


import numpy as np
from libsvm.svmutil import *

TRAIN_NUM = 5000
TEST_NUM = 2500
IMAGE_SIZE = 28
IMAGE_MATRIX = 784


def InputData():
    'Read the data from the file.\nOutput: X and Y for train an test data'
    # train_x: 5000x784 matrix. Every row corresponds to a 28x28 gray-scale image
    train_x = []
    with open('./data/X_train.csv', 'r') as f:
        for i in range(TRAIN_NUM):
            L = f.readline()    # read one line once
            point = L.strip().split(',')    # split ',' of the data for each row
            tmp = [float(p) for p in point]
            train_x.append(tmp)
    # train_y: 5000x1 matrix, which records the class of the training samples
    train_y = []
    with open('./data/Y_train.csv', 'r') as f:
        for i in range(TRAIN_NUM):
            L = f.readline()    # read one line once
            point = L.strip()    # split white space of the data for each row
            train_y.append(float(point))
    # test_x: 2500x784 matrix. Every row corresponds to a 28x28 gray-scale image
    test_x = []
    with open('./data/X_test.csv', 'r') as f:
        for i in range(TEST_NUM):
            L = f.readline()    # read one line once
            point = L.strip().split(',')    # split ',' of the data for each row
            tmp = [float(p) for p in point]
            test_x.append(tmp)
    # test_y: 2500x1 matrix, which records the class of the test samples
    test_y = []
    with open('./data/Y_test.csv', 'r') as f:
        for i in range(TEST_NUM):
            L = f.readline()    # read one line once
            point = L.strip()    # split white space of the data for each row
            test_y.append(float(point))
    return train_x, train_y, test_x, test_y


def LinerKernel(x1, x2):
    'Linear kernel function.\nOutput : kernel value'
    xn = np.array(x1)
    xm = np.array(x2)
    # print(xn[0])
    # print(xm[0])
    # kernel(x, x') = xT * x'
    kernel = np.matmul(xn, xm.T)
    # print(kernel[0])
    return kernel


def RBFKernel(x1, x2, gama):
    'RBF kernel function.\nOutput : kernel value'
    xn = np.array(x1)
    xm = np.array(x2)
    # diff = np.sum(xn ** 2, axis=1).reshape(-1, 1) + np.sum(xm ** 2, axis=1) - 2 * xn @ xm.T
    # ||x-x'||^2 = ||x||^2 - 2*xT*x' + ||x'||^2   -> norm ||x||^2 = sum(xi^2)
    diff = np.sum(xn**2, axis=1).reshape(len(xn), 1) - (2*np.matmul(xn, xm.T)) + np.sum(xm**2, axis=1)
    # kernel(x, x') = exp(-gama * (||x-x'||^2))
    kernel = (-gama) * diff
    kernel = np.exp(kernel)
    return kernel


def GridSearch(mode, X, Y, n):
    'Do the grid search for finding parameters of the best performing model.\nOutput : best parameters'
    best_score = 0
    C_range = [0.001, 0.01, 1, 10]
    gamma_range = [1/IMAGE_MATRIX, 0.01, 0.1, 1]
    coef_range = [0, 1, 2, 3]
    degree_range = [2, 3, 4]
    # grid search start   
    if mode == 0:   # linear -> C
        for c in C_range:
            # for every possible parameter combination, train the model with n cross-validation
            param = f'-t 0 -c {c} -v {n}'
            acc = svm_train(Y, X, param)  # If cross validation is used, either accuracy (ACC) or mean-squared error (MSE) is returned
            # best performing parameters
            if acc > best_score:
                best_score = acc
                best_parameters = f'-t 0 -c {c} -v {n}'
    elif mode == 1: # poly -> C, gamma, e, Q
        for q in degree_range:
            for e in coef_range:
                for gamma in gamma_range:
                    for c in C_range:
                        # for every possible parameter combination, train the model with n cross-validation
                        param = f'-t 1 -d {q} -r {e} -c {c} -v {n}'
                        acc = svm_train(Y, X, param)  # If cross validation is used, either accuracy (ACC) or mean-squared error (MSE) is returned
                        # best performing parameters
                        if acc > best_score:
                            best_score = acc
                            best_parameters = f'-t 1 -d {q} -r {e} -c {c} -v {n}'
    elif mode == 2: # RBF -> C, gamma
        for gamma in gamma_range:
            for c in C_range:
                # for every possible parameter combination, train the model with n cross-validation
                param = f'-t 2 -g {gamma} -c {c} -v {n}'
                acc = svm_train(Y, X, param)  # If cross validation is used, either accuracy (ACC) or mean-squared error (MSE) is returned
                # best performing parameters
                if acc > best_score:
                    best_score = acc
                    best_parameters = f'-t 2 -g {gamma} -c {c} -v {n}'
    return best_parameters
    

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = InputData()

    # Task 1
    # Training model
    model_linear = svm_train(train_Y, train_X, '-t 0')  # linaer kernel
    model_poly = svm_train(train_Y, train_X, '-t 1 -d 2')  # polynomial kernel
    model_RBF = svm_train(train_Y, train_X, '-t 2')  # RBF kernel
    # Prediction
    res_linear = svm_predict(test_Y, test_X, model_linear)  # linaer kernel
    res_poly = svm_predict(test_Y, test_X, model_poly)  # polynomial kernel
    res_RBF = svm_predict(test_Y, test_X, model_RBF)  # RBF kernel

    # Task 2
    # Grid search for the best performing parameters
    n = 3   # n cross-validation
    param = GridSearch(2, train_X, train_Y, n)  # 0 -> linear, 1 -> poly, 2 -> RBF
    best_param = param.replace(f' -v {n}', '')  # remove '-v'
    model_SVC = svm_train(train_Y, train_X, best_param)  # C-SVC train by the best parameters
    res_SVC = svm_predict(test_Y, test_X, model_SVC)    # prediction
    print(f'Best parameter: {best_param}')

    # Task 3
    # Use linear kernel + RBF kernel together, normalize
    gama = 1 / IMAGE_MATRIX
    # Transform training data
    kernel_train = np.zeros((TRAIN_NUM, TRAIN_NUM+1))
    kernel_train[:, 1:] = LinerKernel(train_X, train_X) + RBFKernel(train_X, train_X, gama)
    kernel_train[:, :1] = np.arange(1, TRAIN_NUM+1).reshape(TRAIN_NUM, 1)   # index
    # Transform testing data
    kernel_test = np.zeros((TEST_NUM, TRAIN_NUM+1))
    kernel_test[:, 1:] = LinerKernel(test_X, train_X) + RBFKernel(test_X, train_X, gama)
    kernel_test[:, :1] = np.arange(1, TEST_NUM+1).reshape(TEST_NUM, 1)   # index
    # SVM model
    model_new = svm_train(train_Y, [list(row) for row in kernel_train], '-t 4')  # precomputed kernel : linaer kernel + RBF kernel
    res_new = svm_predict(test_Y, [list(row) for row in kernel_test], model_new)  # linaer kernel + RBF kernel