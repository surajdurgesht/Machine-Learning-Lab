# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:49:51 2019

@author: Suraj
"""

import numpy as np  
import matplotlib.pyplot as plt


def sigmoid(x):  
    temp = 1/(1+np.exp(-x))
    temp = np.append(temp, 1)
    return temp

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

"""
forword feeding
"""
if __name__ == "__main__":
    W1 = np.array([[0.15,0.20,0.35],
                  [0.20,0.30,0.35]])
    IN = np.array([[0.05],[0.10],[1]])
    
    out1 = np.dot(W1,IN)
    print(out1)
    z = sigmoid(out1)
    print(z)
    
    W2 = np.array([[0.45,0.40, 0.6],
                  [0.50,0.55, 0.6]])

    print(z.T)
    net = np.dot(W2,z.T)
    print(net)
    y = sigmoid(net)
    print(y)
    
    t = [0.01,0.99,1]    
    E = np.sum((1/2)*(t-y)**2)
    print(E)
    
"""
Backword propogation
"""

