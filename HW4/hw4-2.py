"""
2022ML homework4-2 created by Pei Hsuan Tsai.
EM algorithm
    Use EM algorithm to cluster each image into ten groups.
"""


import numpy as np
import gzip


#Path of data
PATH = "D:/大四/四下/ML/HW2/"
TrainImageF = "train-images-idx3-ubyte.gz"
TrainLableF = "train-labels-idx1-ubyte.gz"
#Number of example
TrainSet = 60000
ImageSize = 28
Num_Class = 10


def ConfusionMat(i, y_label, y_pred):
    'Create and print the confusion matrix w.\nOutput : the confusion matrix'
    print(f'Confusion Matrix {i}:')
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
    print(f'\t\tPredict number {i}\tPredict not number {i}')
    print(f'Is number {i}\t\t{mat[0][0]}\t\t\t{mat[0][1]}')
    print(f"Is's number {i}\t\t{mat[1][0]}\t\t\t{mat[1][1]}")
    return mat


def Sensitivity(i, mat):
    'Calculate sensitivity of the EM.\nOutput : sensitivity value'
    # sensitivity = TP / (TP + FN)->Yes
    if mat[0][0] + mat[0][1] != 0:
        sensitivity = mat[0][0] / (mat[0][0] + mat[0][1])
    else:
        sensitivity = 0
    print(f'Sensitivity (Successfully predict number {i}): {sensitivity}')
    return


def Specificity(i, mat):
    'Calculate specificity of the EM.\nOutput : specificity value'
    # specificity = TN / (TN + FP)->No
    if mat[1][0] + mat[1][1] != 0:
        specificity = mat[1][1] / (mat[1][0] + mat[1][1])
    else:
        specificity = 0
    print(f'Specificity (Successfully predict number {i}): {specificity}')
    return


def ParseData():
    'Parse the training data.\nOutput: array of train image, train lable'
    #Train Image
    print("Parsing traning image...")
    with gzip.open(PATH+TrainImageF, 'r') as f:
        #First four data are information of image, 4 bytes/each
        for i in range(4):
            f.read(4)
        #60000 examples
        trainIm = []
        for n in range(TrainSet):
            tmpIm = np.zeros([ImageSize, ImageSize], np.uint8) #image size = 28 x 28 x 8bits
            #28x28 matrix, 1 byte/each data, pixel values are 0 to 255
            for i in range(ImageSize):
                for j in range(ImageSize):
                    #Convert bytes to integer (big-endian) => e.g 0x1234 : big->0x12,0x34 ; little->0x34,0x12
                    tmpIm[i][j] = int.from_bytes(f.read(1), byteorder='big', signed=False)
            trainIm.append(tmpIm)

    #Train lable
    print("Parsing traning label...")
    with gzip.open(PATH+TrainLableF, 'r') as f:
        #First two data are information of image, 4 bytes/each
        for i in range(2):
            f.read(4)
        #60000 examples, 1 byte/each data, lable values are 0 to 9
        trainL = []
        for n in range(TrainSet):
            trainL.append(int.from_bytes(f.read(1), byteorder='big', signed=False)) #Convert bytes to integer (big-endian)

    return trainIm, trainL


def AssignLabel(train_L, w):
    'Use training labels to assign which class belongs to which number.\nOutput : assigned label'
    label = np.zeros(Num_Class)
    match = np.zeros((Num_Class, Num_Class))
    # Match the training labels and resposibility w
    for n in range(TrainSet):
        max_w = np.argmax(w[n])
        match[train_L[n]][max_w] += 1
    # Find the best match (max probbility) to assign label
    for k in range(Num_Class):
        # Find the most possible match
        max_match = np.argmax(match)
        classL = max_match // Num_Class
        best_num = max_match % Num_Class
        # The most possible number for class
        label[classL] = best_num
        # Clear the match
        match[classL, :] = 0
        match[:, best_num] = 0

    return label


def ImagePrint(Image, label=None, End=False):
    'Print out the imagination of numbers.\n28x28 binary image which 0 (expect less than 128) represents a white pixel, and 1 (otherwise) represents a black pixel.'
    num = -1
    for n in range(Num_Class):
        if End == True:
            num = label[n]
            print("labeled", end=" ")
        else:
            num += 1
        print(f"class {n}:")

        for i in range(ImageSize*ImageSize):
            # less than value 0.5 -> 0, else 1
            if Image[num][i] < 0.5 :
                print("0", end=" ")
            else:
                print("1", end=" ")
            if i % ImageSize == 27:
                print()
    return


def EM(train_I, train_L):
    'Use EM algorithm to cluster each image into ten groups.'
    # Initial
    iteration = 0
    converge = False
    conv_range = 15
    errorRate = 0
    # probability of appear 0~9 (lembda MLE) -> initial probability are all same
    lembda = np.full((Num_Class), 1/Num_Class)
    # probability of value of every pixel for 0~9 (P MLE) -> random initial
    P = np.ones((Num_Class, ImageSize*ImageSize))
    for n in range(Num_Class):
        for i in range(ImageSize*ImageSize):
            P[n][i] = np.random.uniform(0.2, 0.9)
    # probability of appear 0~9 for each data (resposibility w)
    w = np.zeros((TrainSet, Num_Class))

    # Binning the gray level value into two bins. (value 0~127 -> bin 0; value 128~255 -> bin 1)
    x = np.zeros((TrainSet, ImageSize*ImageSize))
    for n in range(TrainSet):
        for i in range(ImageSize):
            for j in range(ImageSize):
                if train_I[n][i][j] < 128:
                    x[n][i*ImageSize+j] = 0
                else:
                    x[n][i*ImageSize+j] = 1

    # Use EM algorithm to cluster each image into ten groups.
    # Repeat steps untill the lembda and P converage or iteration gets to times limit
    while converge == False and iteration < 10:
        old_P = np.copy(P)
        iteration += 1

        # E step : calculate resposibility w
        # w[i] = lembda[i] * (P[i]^x) * ((1-P[i])^(1-x)) / sum_each_class_k(lembda[k] * (P[k]^x) * ((1-P[k])^(1-x)))
        print("E step...")
        for n in range(TrainSet):
            for i in range(Num_Class):
                tmp = np.zeros(Num_Class)
                for k in range(Num_Class):
                    p1 = P[k] ** x[n]
                    p2 = (1-P[k]) ** (1-x[n])
                    tmp[k] = lembda[k] * np.prod(p1) * np.prod(p2)
                # print(sum(tmp))
                w[n][i] = tmp[i] / sum(tmp)

        # M step : calculate MLE lembda and P
        print("M step...")
        for i in range(Num_Class):
            # lembda[i] = sum(w[i]) / TrainSet
            tmp = sum(w[:, i])
            lembda[i] = tmp / TrainSet
            # P[i] = sum(w[i] * data) / sum(w[i])
            for j in range(ImageSize*ImageSize):
                tmp2 = sum(w[:, i]*x[:, j])
                P[i][j] = tmp2 / tmp

        # Print out the imagination of numbers for each iteration
        ImagePrint(P)
        diff = 0
        for i in range(Num_Class):
            diff += sum(abs(old_P[i] - P[i]))
        print(f"No. of Iteration: {iteration}, Difference: {diff}")
        print("------------------------------------")
        # Calculate that whether the parameters converge, converge range = 15
        if diff <= conv_range:
            converge = True

    # # Print out the labeled imagination
    # Labeled = AssignLabel(train_L, w)
    # ImagePrint(P, Labeled, True)

    # # Prediction
    # y_pred = np.zeros(TrainSet)
    # for i in range(TrainSet):
    #     y_pred[i] = np.argmax(w[i])

    # # Output a confusion matrix, the sensitivity and specificity of the clustering applied to the training data
    # for i in range(Num_Class):
    #     # Print the confusion matrix
    #     confusion = ConfusionMat(i, y_label, y_pred)
    #     # Calculate sensitivity and specificity
    #     Sensitivity(i, confusion)
    #     Specificity(i, confusion)
    #     print("------------------------------------")
    #     errorRate += (confusion[0][1] + confusion[1][0]) / TrainSet

    # errorRate /= Num_Class
    # # Total result
    # print(f'Total iteration to converage:{iteration}')
    # print(f'Total error rate:{errorRate}')
    return


if __name__ == '__main__':
    train_I, train_L = ParseData()
    EM(train_I, train_L)