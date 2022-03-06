# 2022ML homework1 created by Pei Hsuan Tsai
# regularized linear model regression

from cgi import print_directory
import matplotlib.pyplot as plt
import numpy as np

def Input():
    'Input parameters, the path and name of a file which consists of data points (comma seperated: x,y)\nOutput : matrix data, matrix A, matrix b, bases n, lambda l'
    path = input('Inter the path of input file : ')
    name = input('Inter the name of input file : ')

    #get data(x, y)
    data = []
    f = open(path+name, 'r')
    L = f.readlines()
    for line in L:
        point = line.split(',')
        px = float(point[0])
        py = float(point[1])
        point = (px, py)
        data = data + [point]
    f.close()

    n = input('Inter the number of polynomial bases : ')
    l = input('Inter the lambda for LSE : ')
    n = int(n)

    #matrix b = set of y, matrix A = set of x
    data_len = len(data)
    tmp_A = np.zeros([1, data_len])
    A = np.zeros([data_len, n])
    b = np.zeros([data_len, 1])

    for i in range(data_len):
        tmp_A[0][i] = data[i][0]
        b[i][0] = data[i][1]
    for t in range(n):
        for i in range(data_len):
            A[i][n-1-t] = tmp_A[0][i] ** t

    return data, A, b, n, l

def scaleLambda(L, n):
    'scale the n*n identity matrix by lambda L\nOutput : matrix L*I'
    M = np.zeros([n, n])
    for i in range(n):
        M[i][i] = L
    return M

def transpose(A):
    'compute transpose of A\nOutput : transpose of A'
    Arow = len(A)
    Acol = len(A[0])
    At = np.zeros([Acol, Arow])
    for i in range(Acol):
        for j in range(Arow):
            At[i][j] = A[j][i]
    return At

def M_multi(A, B):
    'matrix mutiplication of A and B\nOutput : matrix A*B'
    A_row = len(A)
    A_col = len(A[0])
    B_row = len(B)
    B_col = len(B[0])
    if A_col != B_row:
        raise ValueError("Size of mutiplicated matrix is wrong.")

    multi = np.zeros([A_row, B_col])
    for i in range(A_row):
        for j in range(B_col):
            for t in range(A_col):
                multi[i][j] += (A[i][t] * B[t][j])

    return multi


def LUdecomposition(A):
    'decompose matrix A into L and U, and get inverse of A\nOutput : inverse A'
    n = len(A) #find length of n*n matrix A
    L = np.zeros([n, n]) #initial L by zero
    U = np.zeros([n, n]) #initial L by zero
    #fill value U1i = A1i
    for i in range(n):
        U[0][i] = A[0][i]
    #fill value Li1 = Ai1/U11
    for i in range(n):
        L[i][0] = A[i][0] / U[0][0]
    #fill value Lii = 1
    for i in range(n):
        L[i][i] = 1
    
    for k in range(1, n):
        #fill value Uki = Aki - sum(Lkt*Uti)
        for i in range(k, n):
            U[k][i] = A[k][i]
            for t in range(k):
                U[k][i] -= (L[k][t]*U[t][i])
        #fill value Lik = (Aik - sum(Lit*Utk)) / Ukk
        if k != (n-1): 
            for i in range(k+1, n):
                L[i][k] = A[i][k]
                for t in range(k):
                    L[i][k] -= (L[i][t]*U[t][k])
                L[i][k] /= U[k][k]
    # print('L ', L)
    # print('U ', U)

    #get inverse of L
    L_inverse = np.zeros([n, n])
    for i in range(n):
        L_inverse[i][i] = 1
    for k in range(n-1):
        for i in range(k+1, n):
            for t in range(i):
                L_inverse[i][k] -= (L[i][t]*L_inverse[t][k])
    # print('inverse L ', L_inverse)

    #get inverse of U
    U_inverse = np.zeros([n, n])
    for i in range(n):
        U_inverse[i][i] = 1 / U[i][i]
    for k in range(n-1, 0, -1):
        for i in range(k-1, -1, -1):
            for t in range(i+1, k+1):
                U_inverse[i][k] -= (U[i][t]*U_inverse[t][k])
            U_inverse[i][k] /= U[i][i]
    # print('inverse U ', U_inverse)

    #get inverse of A = inverse of U * inverse of L
    A_inverse = M_multi(U_inverse, L_inverse)

    return A_inverse

def LSE(A, b, L):
    'Use LU decomposition to find the inverse. Calculate x=(AtA+lI)-1*Atb\nOutput : matrix x = (AtA+lI)-1*Atb'
    A_trans = transpose(A)
    tmp_A = M_multi(A_trans, A) + scaleLambda(L, len(A_trans))
    tmp_A = LUdecomposition(tmp_A)
    tmp_B = M_multi(tmp_A, A_trans)
    x = M_multi(tmp_B, b)
    return x

def Newton(A, b):
    'Newton method. Calculate Xi+1 = Xi - (H)-1*gradient for multiple times. H=2AtA, gradient=2AtAx-2Atb\nOutput : min matrix x'
    xi = np.random.randint(1, 10, (len(A[0]), 1)) #x0 initial in random(1, 10)
    A_trans = transpose(A)
    H = 2 * M_multi(A_trans, A)
    G = M_multi(H, xi) - (2 * M_multi(A_trans, b))
    x = xi - M_multi(LUdecomposition(H), G)
    return x

def Function(w, x, n):
    'Use matrix w to make a function, and calculate the f(x)\nOutput : y = f(x)'
    y = 0
    for i in range(n):
        y += w[n-1-i] * (x ** i)
    return y

def Error(data, w, n):
    'Calculate the error of fitting line. error = sum(yi-y)^2\nOutput : value of error'
    error = 0
    for d in data:
        yi = Function(w, d[0], n)
        error += ((yi - d[1]) ** 2)
    error
    return error

def Output(data, x, n):
    'Print output information'
    print("Fitting line:", end=" ")
    for i in range(n):
        if (i+1) != n:
            if x[i] != 0:
                print("%.10fX^" %x[i] + str(n-1-i) + " + ", end="")
        else:
            print("%.10f" %x[i])
    print("Total error: %.10f" %(Error(data, x, n)))
    return

def Visul(data, w1, w2, n):
    'visualize the data points which are the input of program, and the best fitting curve'
    x = []
    y = []
    y1 = []
    y2 = []
    xf = np.linspace(-6, 6, 10000)

    for i in data:
        x = x + [i[0]]
        y = y + [i[1]]

    for i in xf:
        y1 = y1 + [Function(w1, i, n)]
        y2 = y2 + [Function(w2, i, n)]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(x, y)
    ax2.scatter(x, y)
    ax1.plot(xf, y1)
    ax2.plot(xf, y2)
    plt.show()
    return


# n = input('input of matrix n*n')
# D = [[3,5,7,2], [1,4,7,2], [6,3,9,17], [13,5,4,16]]
# print('matrix', D)
# print('inverse', LUdecomposition(D))

#read the data file
D, A, b, n, l = Input()

print()
#use LSE to get fitting line
print("LSE:")
xL = LSE(A, b, l)
Output(D, xL, n)

print()
#use Newton's method to get fitting line
print("Newton's Method:")
xN = Newton(A, b)
Output(D, xN, n)

#visualize the datapoints and fitting curve
Visul(D, xL, xN, n)