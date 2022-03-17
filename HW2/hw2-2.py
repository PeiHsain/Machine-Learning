# 2022ML homework2-2 created by Pei Hsuan Tsai
# Online learning

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
            data.append(line)

    a = input('Inter the parameter a for the initial beta prior : ')
    b = input('Inter the parameter b for the initial beta prior : ')
    return data, a, b

def BetaBinomialConjugation():
    'Use Beta-Binomial conjugation to perform online learning.'
    return

def Output(data, likeli, PriorA, PriorB, PostA, PostB):
    'Print out the Binomial likelihood (based on MLE), Beta prior and posterior probability (parameters only) for each line.'
    for i in range(data.size):
        print("case", i+1, ":", data[i])
        print("Likelihood:", likeli)
        print("Beta prior: a =", PriorA, " b =", PriorB)
        print("Beta posterior: a =", PostA, " b =", PostB)
    return

#Learn the beta distribution of the parameter p (chance to see 1) of the coin tossing trails in batch.
Data, a, b = InputFile()
BetaBinomialConjugation()
Output(Data, likelihood, A_pri, B_pri, A_post, B_post)