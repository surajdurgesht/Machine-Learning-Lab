# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 08:39:02 2019

@author: Suraj
"""
def find_Prob(D, x, y):
    temp=y
    for i in range(3):
        if(D[i]=='0'):
            temp=temp*(1-x)
        if(D[i]=='1'):
            temp=temp*x
        
    return temp
    

import numpy as np

X=np.array([0.1, 0.5, 0.7])
Y=np.array([0.1, 0.1, 0.8])
D=np.array(['000', '001', '010','011', '100','101', '110', '111'])
sump=0
A=np.zeros((8,3), dtype='float64')
for i in range(8):
    for j in range(3):
        A[i][j]=find_Prob(D[i], X[j], Y[j])
        sump=sump+A[i][j]
        print(A[i][j])
    print("\n")

print(sump)

