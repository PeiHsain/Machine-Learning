"""
2022ML homework2-2 created by Pei Hsuan Tsai
Online learning
"""


import numpy as np


def InputFile():
    'Input parameters, the path and name of a file, and parameter a,b for the initial beta prior.\nOutput : data, prior parameter a, prior parameter b'
    path = input('Inter the path of input file : ')
    name = input('Inter the name of input file : ')

    #get data from input file contains many lines of binary outcomes
    data = []
    with open(path+name, 'r') as f:
        L = f.readlines()
        for line in L:
            data.append(line.strip('\n')) # remove the \n on the end of the data line

    a = input('Inter the parameter a for the initial beta prior : ')
    b = input('Inter the parameter b for the initial beta prior : ')
    return data, a, b


def Binomial(MLE, m, N):
    'Compute Likelihood by Binomial distribution. Binomial = (N m) * MLE^m * (1-MLE)^(N-m).\nOutput : likelihood'
    upTmp = 1
    downTmp1 = 1
    downTmp2 = 1
    for i in range(N):
        upTmp *= (i+1)
    for i in range(N-m):
        downTmp1 *= (i+1)
    for i in range(m):
        downTmp2 *= (i+1)
    
    likelihood = (upTmp/(downTmp1*downTmp2)) * (MLE**m) * ((1-MLE)**(N-m))
    return likelihood


def BetaBinomialConjugation(data, a, b):
    'Use Beta-Binomial conjugation to perform online learning.\nOutput : value of likelihood, updated a and b'
    N = len(data)
    m = 0 # how many 1
    for i in data:
        if i == '1':
            m += 1
    MLE = m / N
    likeli = Binomial(MLE, m, N) # calaulate the likelihood
    new_a = m + int(a) # new a = m+a = a of posterior
    new_b = N - m + int(b) # new b = N-m+b = b of posterior
    return likeli, new_a, new_b


def Output(likeli, PriorA, PriorB, PostA, PostB):
    'Print out the Binomial likelihood (based on MLE), Beta prior and posterior probability (parameters only) for each line.'
    print("Likelihood:", likeli)
    print("Beta prior: a =", PriorA, " b =", PriorB)
    print("Beta posterior: a =", PostA, " b =", PostB)
    return


if __name__ == '__main__':
    #Learn the beta distribution of the parameter p (chance to see 1) of the coin tossing trails in batch.
    Data, A_pri, B_pri = InputFile()
    data_len = len(Data)

    for i in range(data_len):
        likelihood, A_post, B_post = BetaBinomialConjugation(Data[i], A_pri, B_pri)
        print("case", i+1, ":", Data[i])
        Output(likelihood, A_pri, B_pri, A_post, B_post)
        # Update the parameters
        A_pri = A_post
        B_pri = B_post
        print()