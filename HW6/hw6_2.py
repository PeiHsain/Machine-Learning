"""
2022ML homework6_2 created by Pei Hsuan Tsai.
Spectral clustering (both normalized cut and ratio cut ).
    Part 1: How the clustering procedure and spectral clustering.
    Part 2: In addition to cluster data into 2 clusters, try more clusters (e.g. 3 or 4 ...) and show your results.
    Part 3: Try different ways to initialize kernel k-means, (e.g. k-means++) and spectral clustering.
    Part 4: For spectral clustering (both normalized cut and ratio cut), try to examine whether the data points within the
            same cluster do have the same coordinates in the eigenspace of graph Laplacian or not.
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

# 100*100 image, each pixel in the image is treated as a data point -> 10000 datapoints
IMG_SIZE = 100
IMG_LENGTH = 10000
COLOR = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [127, 127, 0], [0, 127, 127]]
COLOR_SCATTER = ["red", "blue", "green"]


def InputData():
    'Read two mages from the file.\nOutput: data points of each imges'
    # Read images using matplotlib
    read1 = cv2.imread('image1.png')
    read2 = cv2.imread('image2.png')
    img1 = np.zeros((IMG_LENGTH, 3), dtype=np.float32)
    img2 = np.zeros((IMG_LENGTH, 3), dtype=np.float32)
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # coordinate of data, RGB of data
            img1[int(i*IMG_SIZE+j)] = np.array(read1[i][j], dtype=np.uint32)
            img2[int(i*IMG_SIZE+j)] = np.array(read2[i][j], dtype=np.uint32)
    return img1, img2


def Distance(x1, x2):
    'Calculate the distance between two points.\nOutput : distance'
    x = (x1%IMG_SIZE) - (x2%IMG_SIZE)
    y = (x1//IMG_SIZE) - (x2//IMG_SIZE)
    return (x**2) + (y**2)


def Kernel(x1, x2, gammaS, gammaC):
    'Use defined kernel to compute similarities for two points.\nOutput : kernel value'
    # Multiplying two RBF kernels in order to consider spatial similarity and color similarity at the same time.
    # kernel(x, x') = exp(-gammaS * ||S(x)-S(x')||^2) * exp(-gammaC * ||C(x)-C(x')||^2)
    spatial = -gammaS * Distance(x1[0], x2[0])
    color = -gammaC * np.sum((x1[1] - x2[1]) ** 2)
    kernel = np.exp(spatial) * np.exp(color)
    return kernel


def GramMatrix(data, gammaS, gammaC):
    'Compute Gram matrix by defined kernel.\nOutput : Gram matrix'
    matrix = np.zeros((IMG_LENGTH, IMG_LENGTH), dtype=np.float32)
    for i in range(IMG_LENGTH):
        for j in range(i, IMG_LENGTH):
            matrix[i][j] = Kernel([i, data[i]], [j, data[j]], gammaS, gammaC)
            matrix[j][i] = matrix[i][j]
    return matrix


def DegreeMatrix(w):
    'Compute degree matrix by similarity matrix.\nOutput : degree matrix'
    D = np.zeros((IMG_LENGTH, IMG_LENGTH), dtype=np.float32)
    # Degree dv = sum_u(Wvu)
    d = np.sum(w, axis=1)
    # Degree matrix D is diagonal matrix, Dvu = dv for v=u
    for i in range(IMG_LENGTH):
        D[i][i] = d[i]
    return D


def MinDist(x, centers):
    'Find the min distance to centers.\nOutput : min distance'
    dis = []
    # Compute spatial distance
    for i in range(len(centers)):
        dis.append(np.sum((x-centers[i]) ** 2))
    return np.min(dis)


def InitialMean(mode, k, U):
    'Initial the mean for k-mean by mode. Random mode = 1, ++ mode = 2.\nOutput : k centers'
    # assign = np.zeros((k, IMG_LENGTH))
    C = np.zeros((k, k), dtype=np.float32)
    if mode == 1:
        # Random pick k canters
        center_index = np.random.choice(IMG_LENGTH, k, replace=False)
        for n in range(k):
            C[n] = U[int(center_index[n])]  # row of U
    elif mode == 2:
        # k-mean++
        # random pick first center
        center_index = np.random.randint(low=0, high=IMG_LENGTH)
        C[0] = U[center_index]  # row of U
        # find the farest point to give high probability
        for n in range(1, k):
            dist_sum = 0
            min_dis = np.zeros(IMG_LENGTH, dtype=np.float32)
            for i in range(IMG_LENGTH):
                min_dis[i] = MinDist(U[i], C)
                dist_sum += min_dis[i]
            # randomly pick a value in range sum(min_dist)
            Random = np.random.rand() * dist_sum
            # untill (Random - dist(x)) < 0, x be as new center
            for i in range(IMG_LENGTH):
                Random -= min_dis[i]
                if Random <= 0:
                    C[n] = U[i]
                    break
    return C


def Converge(old_c, new_c):
    'Check if the clustering converge or not.\nOutput : T or F'
    for i in range(IMG_LENGTH):
        if old_c[i] != new_c[i]:
            return False
    return True


def MinDistCluster(x, centers):
    'Find class of the min distance to centers.\nOutput : which cluster'
    dis = []
    # Compute spatial distance
    for i in range(len(centers)):
        dis.append(np.sum((x-centers[i]) ** 2))
    return np.argmin(dis)


def Kmean(k, U, img):
    'Implement kernel k-mean to clustering.\nOutput : final cluster result'
    # Initial centers, random mode = 1, k-mean++ mode = 2
    c = InitialMean(mode=1, k=k, U=U)
    iter = 0
    converge = 0
    # assign = np.copy(init_a)
    cluster = np.zeros(IMG_LENGTH, dtype=np.uint32)
    # Until converge -> converge time = 3
    while converge < 1:
        old_cluster = np.copy(cluster)
        # E-step: find argmin||x - center|| to assign r
        r = np.zeros((k, IMG_LENGTH), dtype=np.uint8)
        for i in range(IMG_LENGTH):
            cluster[i] = MinDistCluster(U[i], c)
            r[int(cluster[i])][i] = 1
        # M-step: update centers ck = sum(rk*x) / sum_(rk)
        for n in range(k):
            r_sum = np.sum(r[n])
            data_for_k = np.zeros(k)
            for i in range(IMG_LENGTH):
                if cluster[i] == n:
                    data_for_k = data_for_k + U[i]
            if r_sum != 0:
                c[n] = data_for_k / r_sum
        # Visualization
        iter += 1
        Visualization(cluster, iter, img)
        # check if it converge
        if Converge(old_cluster, cluster) == True:
            converge += 1
        else:
            converge == 0
    return cluster


def Visualization(cluster, iter, orig_img):
    'Visualize the cluster assignments of data points in each iteration.\nOutput: image of each iteration'
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) # Clustering image
    o_img =  np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)  # Original image
    # Cluster the image
    for i in range(IMG_LENGTH):
        img[int(i//IMG_SIZE)][int(i%IMG_SIZE)] = COLOR[int(cluster[i])]
        o_img[int(i//IMG_SIZE)][int(i%IMG_SIZE)] = orig_img[i]
    # Change BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    o_img = cv2.cvtColor(o_img, cv2.COLOR_RGB2BGR)
    # Plot clustering and origianl images
    fig = plt.figure()
    # Show clustering image 
    fig.add_subplot(1, 2, 1)
    plt.title(f"Spetral Cluster, iteration={iter}")
    plt.imshow(img)
    # Show origianl image
    fig.add_subplot(1, 2, 2)
    plt.title(f"Original image")
    plt.imshow(o_img)

    plt.show()
    fig.savefig(f'./spetral_normal_random_k4_image1/{iter}.png')


def PlotEigenspace(k, U, cluster):
    'Plot the eigenspace of graph Laplacian in 2D or 3D.'
    fig = plt.figure()
    if k == 2:
        x = [[], []]
        y = [[], []]
        plt.title(f"2D eigenspace")
        for i in range(IMG_LENGTH):
            x[int(cluster[i])].append(U[i][0])
            y[int(cluster[i])].append(U[i][1])
        for n in range(k):
            plt.scatter(x[n], y[n], color=COLOR_SCATTER[n], s=0.5)
    elif k == 3:
        x = [[], [], []]
        y = [[], [], []]
        z = [[], [], []]
        plt.title(f"3D eigenspace")
        ax = fig.add_subplot(projection='3d')
        for i in range(IMG_LENGTH):
            x[int(cluster[i])].append(U[i][0])
            y[int(cluster[i])].append(U[i][1])
            z[int(cluster[i])].append(U[i][2])
        for n in range(k):
            ax.scatter(x[n], y[n], z[n], color=COLOR_SCATTER[n], s=0.5)
    plt.show()
    fig.savefig(f'./spetral_normal_random_k4_image1/eigenspace.png')


def NormalizedCut(k, D, W):
    'Normalized cut for spetral cluster.\nOutput : eigenvector matrix, normalized matrix'
    # Calculate Laplacian matrix L = D - W
    L = D - W
    # I = np.identity(IMG_LENGTH)
    # Calculate normalied Laplacian matrix Lsym = D^(-1/2)*L*D^(-1/2)
    D_sqrt = np.sqrt(D)
    for i in range(IMG_LENGTH):
        if D_sqrt[i][i] != 0:
            D_sqrt[i][i] = 1 / D_sqrt[i][i]
    Lsym = np.matmul(np.matmul(D_sqrt, L), D_sqrt)
    # Lsym = I - np.matmul(np.matmul(D_sqrt, W), D_sqrt)
    # Compute eigenvector matrix U(n*k)
    eigenValue, eigenVector = np.linalg.eig(Lsym)
    eigenIndex = np.argsort(eigenValue) # sort eigenvalues
    U = eigenVector[:, eigenIndex[1:(k+1)]]  # except first eigenvector
    # Normalize U as normalized matrix T, tij = uij / sqrt(sum_k(uik^2))
    U_sum = np.sqrt(np.sum((U ** 2), axis=1)).reshape(-1, 1)
    T = U / U_sum
    return U.real, T.real


def RatioCut(k, D, W):
    'Ratio cut for spetral cluster.\nOutput : eigenvector matrix'
    # Calculate Laplacian matrix L = D - W
    L = D - W
    # Compute eigenvector matrix U(n*k)
    eigenValue, eigenVector = np.linalg.eig(L)
    eigenIndex = np.argsort(eigenValue) # sort eigenvalues
    eigenVector = eigenVector[:, eigenIndex]  # except first eigenvector
    U = eigenVector[:, 1:(k+1)].real
    return U


if __name__ == '__main__':
    Image1, Image2 = InputData()
    gammaS = 1 / IMG_LENGTH
    gammaC = 1 / IMG_LENGTH
    K = 4

    # Compute the Gram matrix by kernel and the degree matrix
    print("Compute Gram matrix for image1......")
    W1 = GramMatrix(Image1, gammaS, gammaC)
    D1 = DegreeMatrix(W1)
    print("Compute Gram matrix for image2......")
    W2 = GramMatrix(Image2, gammaS, gammaC)
    D2 = DegreeMatrix(W2)

    # Spectral cluster
    # Normalized cut
    U1, T1 = NormalizedCut(K, D1, W1)   # image1
    U2, T2 = NormalizedCut(K, D2, W2) # image2
    # Ratio cut
    # U1 = RatioCut(K, D1, W1) # image1
    # U2 = RatioCut(K, D2, W2) # image2

    res1 = Kmean(K, T1, Image1)  # image1 for normalized cut
    # res2 = Kmean(K, T2, Image2)  # image2 for normalized cut
    # res1 = Kmean(K, U1, Image1)  # image1 for ratio cut
    # res2 = Kmean(K, U2, Image2)  # image2 for ratio cut
    # Plot the eigenspace of graph Laplacian

    if K < 4:
        PlotEigenspace(K, U1, res1) # image1
        # PlotEigenspace(K, U2, res2) # image2
