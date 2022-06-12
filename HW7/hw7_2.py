"""
2022ML homework7_2 created by Pei Hsuan Tsai.
t-SNE
    Part 1: Modify the code a little bit and make it back to symmetric SNE.
    Part 2: Visualize the embedding of both t-SNE and symmetric SNE.
    Part 3: Visualize the distribution of pairwise similarities in both high-dimensional space and low-dimensional space.\
    Part 4: Play with different perplexity values.
"""
#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
# In[]
import numpy as np
import matplotlib.pyplot as plt


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, mode=1, label=None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
        \nMode 1 = t-SNE, mode 2 = symmetric SNE
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities -> qij: probability in low-D
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if mode == 't_SNE':   # t-SNE
            # t-SNE: qij = (1 + ||yi-yj||^2)^-1 / sum_k((1 + ||yk-yj||^2)^-1)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        elif mode == 'symmetric_SNE': # symmetric SNE
            # symmetric SNE: qij = exp(-||yi-yj||^2) / sum_k(exp(-||yi-yk||^2))
            num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))        
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if mode == 't_SNE':   # t-SNE
                # t-SNE: gradient = 4 * sum_j((pij-qij) * (yi-yj) * (1 + ||yi-yj||^2)^-1)
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            elif mode == 'symmetric_SNE': # symmetric SNE
                # symmetric SNE: gradient = 4 * sum_j((pij-qij) * (yi-yj))
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)  

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            # Visualize current embedding
            EmbeddingVisual(Y, label, mode, iter+1, perplexity)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    # Final result
    EmbeddingVisual(Y, label, mode, iter+1, perplexity)
    # Return solution
    return Y, P, Q


def EmbeddingVisual(Y, label, mode, iter, perplexity):
    'Visualize the embedding of both t-SNE and symmetric SNE.'
    fig = plt.figure(figsize=(10, 10))
    plt.title(f'{mode} Embedding, perplexity={perplexity}, iter={iter}')
    scatter = plt.scatter(Y[:, 0], Y[:, 1], s=20, c=label)
    plt.legend(*scatter.legend_elements(), bbox_to_anchor = (1 , 1))
    plt.show()
    # Save picture
    fig.savefig(f'./{mode}/perplexity{int(perplexity)}/{iter}.jpg')
    

def DistributionVisual(high_d, low_d, mode, perplexity, label):
    'Visualize the distribution of pairwise similarities in both high-dimensional space and low-dimensional space.'
    # Sort data and transform to 
    idx = np.argsort(label)
    sort_high = high_d[:, idx][idx]
    sort_low = low_d[:, idx][idx]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f'{mode} Distribution of Pairwise Similarities, perplexity={perplexity}', fontsize=15)

    # High-dimensional space
    # Plot the heatmap
    im_h = ax[0].imshow(np.log(sort_high))     
    ax[0].set_title('High-dimension')
    ax[0].set_xlabel('sample number')
    ax[0].set_ylabel('sample number')
    # Create colorbar
    cbar = ax[0].figure.colorbar(im_h, ax=ax[0], shrink=0.8)

    # Low-dimensional space
    # Plot the heatmap
    im_l = ax[1].imshow(np.log(sort_low))     
    ax[1].set_title('High-dimension')
    ax[1].set_xlabel('sample number')
    ax[1].set_ylabel('sample number')
    # Create colorbar
    cbar = ax[1].figure.colorbar(im_l, ax=ax[1], shrink=0.8)

    fig.tight_layout()  # keep suitable distance
    plt.show()
    # Save picture
    fig.savefig(f'./{mode}/perplexity{int(perplexity)}/distribution.jpg')


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("./tsne_python/mnist2500_X.txt")
    labels = np.loadtxt("./tsne_python/mnist2500_labels.txt")

    # Hyperparameters
    no_dim = 2
    init_dim = 50
    perplexity = 20.0
    sne_mode = 'symmetric_SNE'    # mode = 't_SNE' or 'symmetric_SNE'

    # Learning
    Y, P, Q = tsne(X, no_dims=no_dim, initial_dims=init_dim, perplexity=perplexity, mode=sne_mode, label=labels)
    
    # Distribution visualization
    DistributionVisual(P, Q, sne_mode, perplexity, labels)
# %%
