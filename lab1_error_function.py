# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:06:41 2019

@author: Suraj
"""

''' import lib '''
import numpy as np
import matplotlib.pyplot as plt

''' create array '''
x = np.array([2,4,6,8])
y = np.array([2,4,6,8])

''' define empty array'''
f = []
g = []
h = []

''' w values in temp '''
for i in range(-10,10): 
    Y = np.multiply(i,x)
    temp = (Y-y)*(Y-y)
    temp1 = abs(Y-y)
    f.append(np.sum(temp))
    h.append(np.sum(temp1))
    g.append(i)

''' ploting graphs '''    
plt.plot(g,f)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.plot(g,h)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('errorfunction1.png')
plt.show()


