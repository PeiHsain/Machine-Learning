"""
2022ML homework5-2 created by Pei Hsuan Tsai.
SVM on MNIST
    Use SVM models to tackle classification on images of hand-written digits.
    Task 1: Use different kernel functions (linear, polynomial, and RBF kernels) and have comparison between their performance.
    Task 2: Please use C-SVC.
    Task 3: Use linear kernel + RBF kernel together (therefore a new kernel function) and compare its performance with respect to others.
"""


import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *

TRAIN_NUM = 5000
TEST_NUM = 2500
IMAGE_SIZE = 28
IMAGE_MATRIX = 784

def InputData():
    'Read the data from the file.\nOutput: X and Y for train an test data'
    # train_x: 5000x784 matrix. Every row corresponds to a 28x28 gray-scale image
    train_x = np.zeros((TRAIN_NUM, IMAGE_MATRIX))
    with open('./data/X_train.csv', 'r') as f:
        for i in range(TRAIN_NUM):
            L = f.readline()    # read one line once
            point = L.split(',')    # split ',' of the data for each row
            for j in range(IMAGE_MATRIX):
                train_x[i][j] = float(point[j])
    # train_y: 5000x1 matrix, which records the class of the training samples
    train_y = np.zeros(TRAIN_NUM)
    with open('./data/Y_train.csv', 'r') as f:
        for i in range(TRAIN_NUM):
            L = f.readline()    # read one line once
            point = L.strip('\n')    # split '\n' of the data for each row
            train_y[i] = float(point)
    # test_x: 2500x784 matrix. Every row corresponds to a 28x28 gray-scale image
    test_x = np.zeros((TEST_NUM, IMAGE_MATRIX))
    with open('./data/X_test.csv', 'r') as f:
        for i in range(TEST_NUM):
            L = f.readline()    # read one line once
            point = L.split(',')    # split ',' of the data for each row
            for j in range(IMAGE_MATRIX):
                test_x[i][j] = float(point[j])
    # test_y: 2500x1 matrix, which records the class of the test samples
    test_y = np.zeros(TEST_NUM)
    with open('./data/Y_test.csv', 'r') as f:
        for i in range(TEST_NUM):
            L = f.readline()    # read one line once
            point = L.strip('\n')    # split '\n' of the data for each row
            test_y[i] = float(point)
    return train_x, train_y, test_x, test_y


def LinerKernel(x1, x2):
    'Linear kernel function.\nOutput : kernel value'
    # kernel(x, x') = xT * x'
    kernel = np.matmul(x1.T, x2)
    return kernel


def PolyKernel(x1, x2, e, gama, Q):
    'Polynomial kernel function.\nOutput : kernel value'
    # kernel(x, x') = (e + gama*xT*x')^Q
    tmp = gama * np.matmul(x1.T, x2)
    kernel = (e + tmp) ** Q
    return kernel


def RBFKernel(x1, x2, gama):
    'RBF kernel function.\nOutput : kernel value'
    # |x-x'|^2 = xT*x - 2*xT*x' + x'T*x'
    diff = np.matmul(x1.T, x1) - (2*np.matmul(x1.T, x2)) + np.matmul(x2.T, x2)
    # kernel(x, x') = exp(-gama * (|x-x'|^2))
    kernel = (-gama) * diff
    kernel = np.exp(kernel)
    return kernel


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = InputData()
    print('1')

    # Task 1

    # Task 2

    # Task 3
