"""
2022ML homework7_1 created by Pei Hsuan Tsai.
Kernel Eigenfaces
    Part 1: Use PCA and LDA to show the first 25 eigenfaces and fisherfaces, and randomly pick 10 images to show their reconstruction.
    Part 2: Use PCA and LDA to do face recognition, and compute the performance.
    Part 3: Use kernel PCA and kernel LDA to do face recognition, and compute the performance.
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path as path

# These data are separated into training dataset(135 images) and testing dataset(30 images).
# 15 subjects (subject01, subject02, etc.) and 11 images per subject.
TRAIN_SET = 135
TEST_SET = 30
SUBJECT = 15
TYPE_TRAIN = 9
TYPE_TEST = 2
TYPE = ["centerlight", "glasses", "happy", "leftlight", "noglasses", "normal", "rightlight", "sad", "sleepy", "surprised", "wink"]
# 32*32 RGB image in 10 classes. 
IMG_SIZE = 32
CLASS = 10


def Read_pgm(pgmf):
    'Return a raster of integers from a PGM as a list of lists.\nOutput : ASCII values of the image'
    assert pgmf.readline() == b'P5\n' # 'P5\n'
    pgmf.readline() # comment line for image's information
    width, height = [int(i) for i in pgmf.readline().split()] # wight and height for the image
    depth = int(pgmf.readline())
    assert depth <= 255 # maximum value
    # read pixel value
    image_value = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            image_value[h][w] = ord(pgmf.read(1))
    return image_value


def InputData():
    'Read training and testing images from the file.\nOutput: training and testing datas'
    # Read training images
    training = []
    FILE_PATH = "./Yale_Face_Database/Training/"
    for i in range(1, SUBJECT+1):
        for j in range(len(TYPE)):
            FILE_NAME = f"subject{i:02}.{TYPE[j]}.pgm"
            if path.exists(FILE_PATH + FILE_NAME) == True:
                imagef = open(FILE_PATH + FILE_NAME, 'rb')
                training.append(Read_pgm(imagef))
    # Read testing images
    testing = []
    FILE_PATH = "./Yale_Face_Database/Testing/"
    for i in range(1, SUBJECT+1):
        for j in range(len(TYPE)):
            FILE_NAME = f"subject{i:02}.{TYPE[j]}.pgm"
            if path.exists(FILE_PATH + FILE_NAME) == True:
                imagef = open(FILE_PATH + FILE_NAME, 'rb')
                testing.append(Read_pgm(imagef))
    return training, testing


def LinerKernel(x1, x2):
    'Linear kernel function.\nOutput : kernel value'
    # kernel(x, x') = xT * x'
    kernel = np.matmul(x1, x2.T)
    return kernel


def PolyKernel(x1, x2, gamma, coef, degree):
    'Polynomial kernel function.\nOutput : kernel value'
    # kernel(x, x') = (gamma * (xT * x') + coef)^degree
    kernel = gamma * np.matmul(x1, x2.T)
    kernel = (kernel + coef) ** degree
    return kernel


def RBFKernel(x1, x2, gama):
    'RBF kernel function.\nOutput : kernel value'
    # diff = np.sum(xn ** 2, axis=1).reshape(-1, 1) + np.sum(xm ** 2, axis=1) - 2 * xn @ xm.T
    # ||x-x'||^2 = ||x||^2 - 2*xT*x' + ||x'||^2   -> norm ||x||^2 = sum(xi^2)
    diff = np.sum(x1**2, axis=1).reshape(len(x1), 1) - (2*np.matmul(x1, x2.T)) + np.sum(x2**2, axis=1)
    # kernel(x, x') = exp(-gama * (||x-x'||^2))
    kernel = (-gama) * diff
    kernel = np.exp(kernel)
    return kernel


def Reconstruction():
    'show the first 25 eigenfaces and fisherfaces, and randomly pick 10 images to show their reconstruction.'


def PCA():
    'Use PCA to do face recognition.'


def KernelPCA():
    'Use kernel PCA to do face recognition.'


def LDA():
    'Use LDA to do face recognition.'


def KernelLDA():
    'Use kernel LDA to do face recognition.'


if __name__ == '__main__':
    train_set, test_set = InputData()
