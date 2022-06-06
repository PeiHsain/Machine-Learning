"""
2022ML homework7_1 created by Pei Hsuan Tsai.
Kernel Eigenfaces
    Part 1: Use PCA and LDA to show the first 25 eigenfaces and fisherfaces, and randomly pick 10 images to show their reconstruction.
    Part 2: Use PCA and LDA to do face recognition, and compute the performance.
    Part 3: Use kernel PCA and kernel LDA to do face recognition, and compute the performance.
"""


# In[]
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

# These data are separated into training dataset(135 images) and testing dataset(30 images).
# 15 subjects (subject01, subject02, etc.) and 11 images per subject.
TRAIN_SET = 135
TEST_SET = 30
SUBJECT = 15
TYPE_TRAIN = 9
TYPE_TEST = 2
TYPE = ["centerlight", "glasses", "happy", "leftlight", "noglasses", "normal", "rightlight", "sad", "sleepy", "surprised", "wink"]
# Face images (width=195, height=231). 
WIDTH = 195
HEIGHT = 231


def Read_pgm(pgmf):
    'Return a raster of integers from a PGM as a list of lists.\nOutput : ASCII values of the image'
    assert pgmf.readline() == b'P5\n' # 'P5\n'
    pgmf.readline() # comment line for image's information
    WIDTH, HEIGHT = [int(i) for i in pgmf.readline().split()] # wight and height for the image
    depth = int(pgmf.readline())
    assert depth <= 255 # maximum value
    # read pixel value
    image_value = np.zeros((HEIGHT, WIDTH))
    for h in range(HEIGHT):
        for w in range(WIDTH):
            image_value[h][w] = ord(pgmf.read(1))
    return image_value.reshape(-1)


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


# In[]
def LinerKernel(x1, x2):
    'Linear kernel function.\nOutput : kernel value'
    # kernel(x, x') = xT * x'
    kernel = np.matmul(x1, x2.T)
    return kernel


def PolyKernel(x1, x2, gamma=0.01, coef=0.1, degree=2):
    'Polynomial kernel function.\nOutput : kernel value'
    # kernel(x, x') = (gamma * (xT * x') + coef)^degree
    kernel = gamma * np.matmul(x1, x2.T)
    kernel = (kernel + coef) ** degree
    return kernel


def RBFKernel(x1, x2, gamma=1e-11):
    'RBF kernel function.\nOutput : kernel value'
    # diff = np.sum(xn ** 2, axis=1).reshape(-1, 1) + np.sum(xm ** 2, axis=1) - 2 * xn @ xm.T
    # ||x-x'||^2 = ||x||^2 - 2*xT*x' + ||x'||^2   -> norm ||x||^2 = sum(xi^2)
    diff = np.sum(x1**2, axis=1).reshape(-1, 1) - (2*np.matmul(x1, x2.T)) + np.sum(x2**2, axis=1)
    # kernel(x, x') = exp(-gama * (||x-x'||^2))
    kernel = (-gamma) * diff
    kernel = np.exp(kernel)
    return kernel


def Kernal(x1, x2, mode):
    'Compute kernel values accroding to the mode.\nOutput : kernel value'
    if mode == 1:
        print('Use linear kernel')
        kernel = LinerKernel(x1, x2)
    elif mode == 2:
        print('Use polynomial kernel')
        kernel = PolyKernel(x1, x2)
    elif mode == 3:
        print('Use RBF kernel')
        kernel = RBFKernel(x1, x2)
    return kernel   


def Reconstruction(eigen, mean, x, save_path, k, n):
    'show the first k(25) eigenfaces, and randomly pick n(10) images to show their reconstruction.'
    size = np.sqrt(k)
    # Show and save eigenface
    fig = plt.figure(figsize=(11, 11))
    plt.title('25 Eigenface')
    for i in range(k):
        eigenface = eigen[:, i].reshape((HEIGHT, WIDTH))
        ax = fig.add_subplot(int(size), int(size), i+1)
        ax.imshow(eigenface, cmap='gray')
    FILE_NAME = f'{k}eigenface.jpg'
    fig.savefig(save_path + FILE_NAME)
    plt.show()

    # Reconstruction face
    fig = plt.figure(figsize=(9, 11))
    plt.title('Randomly 10 reconstruction')
    random_num = np.random.choice(TRAIN_SET, size=n, replace=False)
    for i in range(n):
        # original face
        originalface = x[random_num[i]].reshape((HEIGHT, WIDTH))
        ax = fig.add_subplot(2*2, int(n/2), i+1)
        ax.set_title(f'Original {random_num[i]}')
        ax.imshow(originalface, cmap='gray')

        # reconstruction face
        ax = fig.add_subplot(2*2, int(n/2), (i+1)+n)
        ax.set_title(f'Reconstruct {random_num[i]}')
        # reconstruction = meanface + (x - mean) * W * WT -> 1*N
        reconstruct = mean + np.matmul(np.matmul((x[random_num[i]]-mean), eigen), eigen.T)
        reconstruvtface = reconstruct.reshape((HEIGHT, WIDTH))
        ax.imshow(reconstruvtface, cmap='gray')
    FILE_NAME = f'{n}reconstructface.jpg'
    fig.savefig(save_path + FILE_NAME)
    plt.show()


def PCA(x, k=25):
    'Use PCA to do face recognition.\nOutput : eigenvector matrix, meanface'
    # Mean x, sum each dimension values -> 1*N array, meanface
    mean = np.mean(x, axis=0)
    # Covariance C = (x-x_mean) * (x-x_mean)T
    C = np.matmul((x-mean), (x-mean).T)
    # Orthogonal projection W = k first largest eigenvectors
    eigenValue, eigenVector = np.linalg.eig(C)
    eigenIndex = np.argsort(-eigenValue) # sort eigenvalues
    # Normalize eigenvectors
    eigenVector = np.matmul(x.T, eigenVector)
    norm_eigen = np.linalg.norm(eigenVector, axis=0)
    for i in range(len(norm_eigen)):
        eigenVector[:, i] /= norm_eigen[i]
    W = eigenVector[:, eigenIndex[:k]].real # k first eigenvectors
    return W, mean


def KernelPCA(x, k=25, mode=1):
    'Use kernel PCA to do face recognition. mode=1: linear, mode=2: poly, mode=3: RBF\nOutput : eigenvector matrix, kernel_x'
    # Kernel x -> mode=1: linear, mode=2: poly, mode=3: RBF
    kernel = Kernal(x, x, mode)
    # Center kernel data KC = K - array_N(1/N) * K - K * array_N(1/N) + array_N(1/N) * K * array_N(1/N)
    matrix_1N = np.full((TRAIN_SET, TRAIN_SET), 1/TRAIN_SET)
    KC = kernel - matrix_1N @ kernel - kernel @ matrix_1N + matrix_1N @ kernel @ matrix_1N
    # Orthogonal projection W = k first largest eigenvectors
    eigenValue, eigenVector = np.linalg.eig(KC)
    eigenIndex = np.argsort(-eigenValue) # sort eigenvalues
    # Normalize eigenvectors
    norm_eigen = np.linalg.norm(eigenVector, axis=0)
    for i in range(len(norm_eigen)):
        eigenVector[:, i] /= norm_eigen[i]
    W = eigenVector[:, eigenIndex[:k]].real # k first eigenvectors
    return W, kernel


def LDA():
    'Use LDA to do face recognition.\nOutput : eigenvector matrix, meanface'


def KernelLDA():
    'Use kernel LDA to do face recognition. mode=1: linear, mode=2: poly, mode=3: RBF\nOutput : eigenvector matrix, kernel_x'


def Distance(train, test):
    'Compute distance between test and train data.\nOutput : array of distance'
    dist = np.zeros(TRAIN_SET, dtype=np.float32)
    for i in range(TRAIN_SET):
        # Euclidean distance
        dist[i] = distance.euclidean(train[i], test)
    return dist


def Recognition(train, test, eigen, mean, knn=5):
    'Face recognition. Use k nearest neighbor to classify which subject.\nOutput : the prediction'
    # Project test and train data into eigenvectors
    train_proj = np.matmul((train-mean), eigen)
    test_proj = np.matmul((test-mean), eigen)

    # K nearest neighbor to classify
    prediction = np.zeros(TEST_SET, dtype=np.uint32)
    for i in range(TEST_SET):
        dist = Distance(train_proj, test_proj[i])
        nearest_dist = np.argsort(dist)[:knn]
        vote = np.zeros(SUBJECT, dtype=np.uint32)
        for j in range(knn):
            pred_subject = nearest_dist[j] // TYPE_TRAIN
            vote[pred_subject] += 1
        prediction[i] = np.argmax(vote) + 1
    return prediction


def KernelRecognition(train, test, eigen, train_kernel, k_mode, knn=5):
    'Face recognition. Use k nearest neighbor to classify which subject.\nOutput : the prediction'
    # Project kernel test and kernel train data into eigenvectors
    train_proj = np.matmul(train_kernel, eigen)
    # Center test kernel values
    test_kernel = Kernal(test, train, k_mode)
    matrix_1N_test = np.full((TEST_SET, TEST_SET), 1/TEST_SET)
    matrix_1N_train = np.full((TRAIN_SET, TRAIN_SET), 1/TRAIN_SET)
    KC = test_kernel - matrix_1N_test @ test_kernel - test_kernel @ matrix_1N_train + matrix_1N_test @ test_kernel @ matrix_1N_train
    test_proj = np.matmul(test_kernel, eigen)

    # K nearest neighbor to classify
    prediction = np.zeros(TEST_SET, dtype=np.uint32)
    for i in range(TEST_SET):
        dist = Distance(train_proj, test_proj[i])
        nearest_dist = np.argsort(dist)[:knn]
        vote = np.zeros(SUBJECT, dtype=np.uint32)
        for j in range(knn):
            pred_subject = nearest_dist[j] // TYPE_TRAIN
            vote[pred_subject] += 1
        prediction[i] = np.argmax(vote) + 1
    return prediction


def Accuracy(pred):
    'Compute the performance of face recognition.'
    error = 0
    for i in range(TEST_SET):
        truth_subj = (i // 2) + 1
        if pred[i] != truth_subj:
            error += 1
    acc = 1 - (error / TEST_SET)
    print(f'Accuracy = {acc}')



# In[]
if __name__ == '__main__':
    train_set, test_set = InputData()
    train_set = np.array(train_set)
    test_set = np.array(test_set)
# In[]
    K = 25
    N = 10
    KNN = 5
    # mode=1: linear kernel, mode=2: poly kernel, mode=3: RBF kernel
    K_mode = 2

# In[]
    # PCA
    FILE_PATH = './PCA/'
    pca_W, pca_mean = PCA(train_set)
    # Reconstruction(pca_W, pca_mean, train_set, FILE_PATH, K, N)
    pca_pred = Recognition(train_set, test_set, pca_W, pca_mean, KNN)
    print('PCA method performance :')
    Accuracy(pca_pred)

# In[]
    # LDA
    FILE_PATH = './LDA/'
    lda_W, lda_mean = LDA(train_set)
    Reconstruction(lda_W, lda_mean, train_set, FILE_PATH, K, N)
    lda_pred = Recognition(train_set, test_set, lda_W, lda_mean, KNN)
    print('LDA method performance :')
    Accuracy(lda_pred)

# In[]
    # Kernel PCA
    kpca_W, kpca_k = KernelPCA(train_set, K, K_mode)
    kpca_pred = KernelRecognition(train_set, test_set, kpca_W, kpca_k, K_mode, KNN)
    print('Kernel PCA method performance :')
    Accuracy(kpca_pred)

# In[]
    # Kernel LDA
    klda_W, klda_mean = KernelLDA(train_set)
    klda_pred = Recognition(train_set, test_set, klda_W, klda_mean, KNN)
    print('Kernel LDA method performance :')
    Accuracy(klda_pred)

# %%
