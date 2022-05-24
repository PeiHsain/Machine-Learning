"""
2022ML homework6_1 created by Pei Hsuan Tsai.
Kernel k-means
Spectral clustering (both normalized cut and ratio cut ).
    Part 1: How the clustering procedure and spectral clustering.
    Part 2: In addition to cluster data into 2 clusters, try more clusters (e.g. 3 or 4 ...) and show your results.
    Part 3: Try different ways to initialize kernel k-means, (e.g. k-means++) and spectral clustering.
    Part 4: For spectral clustering (both normalized cut and ratio cut), try to examine whether the data points within the
            same cluster do have the same coordinates in the eigenspace of graph Laplacian or not.
"""

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
# from scipy.spatial.distance import dis

# 100*100 image, each pixel in the image is treated as a data point -> 10000 datapoints
IMG_SIZE = 100
IMG_LENGTH = 10000
COLOR = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [127, 127, 0], [0, 127, 127]]


def InputData():
    'Read two mages from the file.\nOutput: data points of each imges'
    # Read images using matplotlib
    read1 = cv2.imread('image1.png')
    read2 = cv2.imread('image2.png')
    img1 = []
    img2 = []
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # coordinate of data, RGB of data
            img1.append((np.array([j, i]), read1[i][j]))
            img2.append((np.array([j, i]), read2[i][j]))
    return img1, img2


def Distance(x1, x2):
    'Calculate the distance between two points.\nOutput : distance'
    dis = np.sum((x1 - x2) ** 2)
    return dis


def Kernel(x1, x2, gammaS, gammaC):
    'Use defined kernel to compute similarities for two points.\nOutput : kernel value'
    # Multiplying two RBF kernels in order to consider spatial similarity and color similarity at the same time.
    # kernel(x, x') = exp(-gammaS * ||S(x)-S(x')||^2) * exp(-gammaC * ||C(x)-C(x')||^2)
    spatial = -gammaS * Distance(x1[0], x2[0])
    color = -gammaC * Distance(x1[1], x2[1])
    kernel = np.exp(spatial) * np.exp(color)
    return kernel


def GramMatrix(data, gammaS, gammaC):
    'Compute Gram matrix by defined kernel.\nOutput : Gram matrix'
    matrix = np.zeros((IMG_LENGTH, IMG_LENGTH))
    for i in range(IMG_LENGTH):
        for j in range(i, IMG_LENGTH):
            matrix[i][j] = Kernel(data[i], data[j], gammaS, gammaC)
            matrix[j][i] = matrix[i][j]
    return matrix


# In[2]:

def MinDist(index, centers):
    'Find the min distance to centers.\nOutput : min distance'
    min_dist = 1000000 
    x = index % IMG_SIZE
    y = index // IMG_SIZE
    # Compute spatial distance
    for i in range(len(centers)):
        c_x = centers[i] % IMG_SIZE
        c_y = centers[i] // IMG_SIZE
        dis = (x-c_x) ** 2 + (y-c_y) ** 2
        if dis < min_dist:
            min_dist = dis
    return min_dist


def InitialMean(mode, k):
    'Initial the mean for k-mean by mode. Random mode = 1, ++ mode = 2.\nOutput : initial C and a of k centers'
    assign = np.zeros((k, IMG_LENGTH))
    C = np.zeros(k)
    if mode == 1:
        # Random pick k canters
        center_index = np.random.choice(IMG_LENGTH, k, replace=False)
        # print(index)
        for n in range(k):
            assign[n][center_index[n]] = 1
    elif mode == 2:
        # k-mean++
        center_index = np.zeros(k)
        # random pick first center
        center_index[0] = np.random.randint(low=0, high=IMG_LENGTH)
        assign[0][int(center_index[0])] = 1
        # find the farest point to give high probability
        for n in range(1, k):
            dist_sum = 0
            min_dis = np.zeros(IMG_LENGTH)
            for i in range(IMG_LENGTH):
                min_dis[i] = MinDist(i, center_index)
                dist_sum += min_dis[i]
            # randomly pick a value in range sum(min_dist)
            Random = np.random.rand() * dist_sum
            # untill (Random - dist(x)) < 0, x be as new center
            for i in range(IMG_LENGTH):
                Random -= min_dis[i]
                if Random < 0:
                    center_index[n] = i
                    assign[n][int(center_index[n])] = 1
                    break
    for n in range(k):
        C[n] = np.sum(assign[n])
    return C, assign


def Converge(old_assign, new_assign):
    'Check if the clustering converge or not.\nOutput : T or F'
    for i in range(IMG_LENGTH):
        if np.argmin(old_assign[:, i]) != np.argmin(new_assign[:, i]):
            return False
    return True


def KernelDist(k, kernel, c, a):
    'Distance between data and center of kernel k-mean.\nOutput : array of distance'
    # kernel(x, x) = 1
    dis = np.ones((k, IMG_LENGTH))
    for n in range(k):
        # term 2 = -2 * sum_i(an*kernel(i, all)) / cn
        alpha = a[n].reshape(IMG_LENGTH, 1)
        term2 = np.matmul(alpha.T, kernel)
        term2 = 2 * term2 / c[n]
        dis[n] -= term2.flatten()
        # term 3 = sum_p(sum_q(anp*anq*kernel(p, q))) / cn
        term3 = np.matmul(np.matmul(alpha.T, kernel), alpha)
        term3 /= c[n] ** 2
        dis[n] += term3.flatten()
    return dis     
    

def Kmean(k, init_c, init_a, kernel):
    'Implement kernel k-mean to clustering.'
    iter = 0
    converge = 0
    c = np.copy(init_c)
    assign = np.copy(init_a)
    cluster = np.zeros(IMG_LENGTH)
    # Until converge -> converge time = 3
    while converge < 3:
        old_assign = np.copy(assign)
        # Distance of all points
        dist = KernelDist(k, kernel, c, assign)
        # Find min similarity to assign
        c = np.zeros(k)
        assign = np.zeros((k, IMG_LENGTH))
        for i in range(IMG_LENGTH):
            cluster[i] = np.argmin(dist[:, i])
            assign[int(cluster[i])][i] = 1
        for n in range(k):
            c[n] = np.sum(assign[n])
        print(c)
        # Visualization
        iter += 1
        Visualization(cluster, iter)
        # check if it converge
        if Converge(old_assign, assign) == True:
            converge += 1
        else:
            converge == 0


def Visualization(cluster, iter):
    'Visualize the cluster assignments of data points in each iteration.\nOutput: image of each iteration'
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Cluster the image
    for i in range(IMG_LENGTH):
        img[int(i//IMG_SIZE)][int(i%IMG_SIZE)] = COLOR[int(cluster[i])]
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Show image
    plt.title(f"Kernel K-Mean, iteration={iter}")
    plt.imshow(img)
    plt.show()
    # plt.imsave(f'./kernel_random_k2/{iter}.png', img)

# In[2]:



if __name__ == '__main__':
    Image1, Image2 = InputData()
    gammaS = 1 / IMG_LENGTH
    gammaC = 1 / 256
    K = 2

# In[2]:


    # Compute the Gram matrix by kernel
    print("Compute Gram matrix......")
    gram_matrix = GramMatrix(Image1, gammaS, gammaC)

# In[2]:

    # Kernel K-mean
    # Initial, random mode = 1, k-mean++ mode = 2
    init_C, init_a = InitialMean(mode=1, k=K)
    Kmean(K, init_C, init_a, gram_matrix)

# %%
