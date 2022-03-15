# 2022ML homework2-1 created by Pei Hsuan Tsai
# Naive Bayes classifier

from itertools import count
from pickle import FALSE, TRUE
from unicodedata import category
import numpy as np
import gzip
import math

#Path of data
PATH = "D:/大四/四下/ML/HW2/"
TrainImageF = "train-images-idx3-ubyte.gz"
TrainLableF = "train-labels-idx1-ubyte.gz"
TestImageF = "t10k-images-idx3-ubyte.gz"
TestLableF = "t10k-labels-idx1-ubyte.gz"
#Number of example
TrainSet = 60000
TestSet = 10000
ImageSize = 28

def ParseData():
    'Parse training and testing data.\nOutput: array of train image, train lable, test image, test lable'

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


    #Test Image
    print("Parsing testing image...")
    with gzip.open(PATH+TestImageF, 'r') as f:
        #First four data are information of image, 4 bytes/each
        for i in range(4):
            f.read(4)
        #10000 examples
        testIm = []
        for n in range(TestSet):
            tmpIm = np.zeros([ImageSize, ImageSize], np.uint8) #image size = 28 x 28 x 8bits
            #28x28 matrix, 1 byte/each data, pixel values are 0 to 255
            for i in range(ImageSize):
                for j in range(ImageSize):
                    #Convert bytes to integer (big-endian) => e.g 0x1234 : big->0x12,0x34 ; little->0x34,0x12
                    tmpIm[i][j] = int.from_bytes(f.read(1), byteorder='big', signed=False)
            testIm.append(tmpIm)

    #Test lable
    print("Parsing testing label...")
    with gzip.open(PATH+TestLableF, 'r') as f:
        #First two data are information of image, 4 bytes/each
        for i in range(2):
            f.read(4)
        #10000 examples, 1 byte/each data, lable values are 0 to 9
        testL = []
        for n in range(TestSet):
            testL.append(int.from_bytes(f.read(1), byteorder='big', signed=False)) #Convert bytes to integer (big-endian)


    return trainIm, trainL, testIm, testL

def PosteriorPrint(Posterior, label):
    'Print out the the posterior (in log scale to avoid underflow) of the ten categories (0-9) for each image in INPUT 3. Print out the prediction which is the category and correct answer (lable).\nReturn prediction correct(0) or not(1).'
    print("Posterior (in log scale):")
    for i in range(10):
        print(i, ": ", Posterior[i])
    #Returns the indices of the maximum values
    prediction = np.argmax(Posterior)
    print("Prediction: ", prediction, end="")
    print(", Ans: ", label)

    #correct of prediction
    if prediction == label: #correct
        return 0
    else: #error
        return 1

def ImagePrint(Image):
    'Print out the imagination of numbers in Bayes classifier.\n28x28 binary image which 0 (expect less than 128) represents a white pixel, and 1 (otherwise) represents a black pixel.'
    print("Imagination of numbers in Bayes classifier:")
    for n in range(10):
        print(n, ":")
        for i in range(ImageSize):
            for j in range(ImageSize):
                bin = np.argmax(Image[n][i*ImageSize+j])
                if bin < 16 : #less than value 128 = less than bin 16
                    print("0", end=" ")
                else:
                    print("1", end=" ")
            print()


    return

def Discrete(train_I, train_L, test_I, test_L):
    'Tally the frequency of the values of each pixel into 32 bins.'
    categoryP = np.zeros(10) #save count of categories 0-9
    pixelBin = np.ones([10, ImageSize*ImageSize, 32]) #save count of each category of each unit of pixel bin, avoid empty bin
    
    print("Discrete Mode :")

    #training model, calculate distribute of pixel values for each category
    for n in range(TrainSet):
        categoryP[train_L[n]] += 1 #which category
        for i in range(ImageSize):
            for j in range(ImageSize):
                bin = train_I[n][i][j] // 8 #8 values each bin => values 0-255 for 32 bins
                pixelBin[train_L[n]][i*ImageSize+j][bin] += 1 #which bin
    categoryP /= TrainSet #probability of categories

    #testing model, calculate posterior of each category for each sample, log(post)=log(likelihood)+log(prior)
    posterior = np.zeros(10)
    error = 0
    for n in range(TestSet):
        print("Data", n+1)
        postSum = 0
        for c in range(10):
            posterior[c] = math.log(categoryP[c]) #log(prior)
            for i in range(ImageSize):
                for j in range(ImageSize):
                    bin = test_I[n][i][j] // 8 #8 values each bin => values 0-255 for 32 bins
                    pixSum = pixelBin[c][i*ImageSize+j][bin]
                    posterior[c] += math.log(pixSum / categoryP[c]) #log(likelihood)=log(pixelSum of category / prior)
            postSum += posterior[c]
        #marginalize them so sum it up will equal to 1
        posterior /= postSum
        #print each posterior, prediction and answer
        error += PosteriorPrint(posterior, test_L[n])
    #print my Bayes classifier image 0-9
    ImagePrint(pixelBin)

    #Calculate and print the error rate
    error /= TestSet
    print("Error rate: ", error)
    return

def Continuous(train_I, train_L, test_I, test_L):
    'Use MLE to fit a Gaussian distribution for the value of each pixel.'
    return






#INPUT 1: Training image data
#INPUT 2: Training lable data
#INPUT 3: Testing image
#INPUT 4: Testing label
Im_train, L_train, Im_test, L_test = ParseData()
#INPUT 5: Toggle option, 0 for discrete, 1 for continuous
m = input("Which mode(0 -> discrete, 1 -> continuous): ")

#which mode
if m == '0':
    Discrete(Im_train, L_train, Im_test, L_test)
elif m == '1':
    Continuous(Im_train, L_train, Im_test, L_test)
else:
    print("Wrang mode selection.")