# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:35:14 2019

@author: Suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

x = np.linspace(-1, 1, 100)
sinCurve = np.sin(pi * x)

plt.plot(x, sinCurve)

x_val = []
y_val = []

temp_x = np.linspace(-1, 1, 100)
def hypo1(p1, p2):
    mean = (p1[1] + p2[1]) / 2
    slope = 0
    y_1 = slope * temp_x + mean  
    plt.plot(temp_x, y_1, color = 'k', alpha = 0.2)
    bias1 = np.mean((y_1 - sinCurve)**2)
    bias1 = np.mean(bias1)
    var1 = np.mean((y_1 - bias1)**2)
    return bias1, var1    

temp = []
temp_x2 = np.linspace(-1, 1, 100)
def hypo2(p1, p2, slope):    
    c = p1[1] - (p1[0] * slope)
    y_2 = slope * temp_x2 + c
    axes = plt.gca()
    axes.set_ylim([-1, 1])
    plt.plot(x, sinCurve, color = 'r')
    plt.plot(temp_x2, y_2, color = 'k', alpha = 0.2)  
    bias2 = np.mean((y_2 - sinCurve)**2)
    bias2 = np.mean(bias2)
    var2 = np.mean((y_2 - bias2)**2)
    return bias2,var2
    
def calcBias(temp_x, y_1):
    print(temp_x)

data = []

for i in range(100):
    x1 = np.random.choice(x)
    y1 = np.sin(pi * x1)
    
    x2 = np.random.choice(x)
    y2  = np.sin(pi * x2)
    
    data.append([[x1, y1], [x2, y2]])

    
for point in (data):
    p1 = point[0]
    p2 = point[1]
    b1,v1 = hypo1(p1, p2)
    
print("Bias1: ", b1)
print("Variance: ",v1)
plt.show()

slopes = []
for point in (data):
    p1 = point[0]
    p2 = point[1]
    slope = ((p2[1] - p1[1]) / (p2[0] - p1[0]))
    b2,v2 = hypo2(p1, p2, slope)
    
print("Bias2: ", b2)
print("Variance: ",v2)
    
#print(slopes)
plt.show()
